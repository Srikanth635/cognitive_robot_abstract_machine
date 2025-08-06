import os
import random
import tarfile
from dataclasses import dataclass
from typing import List, Optional
from urllib.parse import urljoin

import rclpy
import requests
from bs4 import BeautifulSoup
from sqlalchemy import create_engine, select, exists
from sqlalchemy.orm import Session

from semantic_world.geometry import Mesh
from semantic_world.orm.ormatic_interface import *

from semantic_world.adapters.urdf import URDFParser
from semantic_world.world import World


@dataclass
class ProcTHORInterface:
    """
    Interface for interacting with the ProcThor environments.
    This class provides methods to scrape, download, and extract .tar.gz files containing ProcThor environments.
    Base URL defaults to 'https://user.informatik.uni-bremen.de/~luc_kro/procthor_environments/'
    """

    base_url: str = "https://user.informatik.uni-bremen.de/~luc_kro/procthor_environments/"
    """
    The base URL to scrape for .tar.gz files containing ProcThor environments.
    """

    project_root: str = os.path.join(os.path.expanduser("~"), "semantic_world")
    """
    The root path of the project, used to find resources and other files.
    """

    def get_tarball_links(self) -> List[str]:
        """
        Scrape and return all .tar.gz file links from the specified base URL.

        :return: A list of URLs pointing to .tar.gz files.
        """
        response = requests.get(self.base_url)
        soup = BeautifulSoup(response.text, "html.parser")
        return [
            urljoin(self.base_url, a["href"])
            for a in soup.find_all("a", href=True)
            if a["href"].endswith(".tar.gz")
        ]

    def download_file(self, url: str, filename: str):
        """
        Download a file from a URL to a destination path.

        :param url: The URL of the file to download.
        :param filename: The name of the file to save as.
        """
        response = requests.get(url, stream=True)
        tmp_folder = os.path.join(self.project_root, "tmp")
        os.makedirs(tmp_folder, exist_ok=True)
        with open(os.path.join(tmp_folder, filename), "wb") as f:
            for chunk in response.iter_content(chunk_size=8192):
                if chunk:
                    f.write(chunk)

    def extract_tar(self, filename: str, extract_to: str):
        """
        Extract a .tar.gz archive to the specified directory.

        :param filename: The name of the .tar.gz file to extract.
        :param extract_to: Directory where the contents should be extracted.
        """
        os.makedirs(extract_to, exist_ok=True)
        mode = "r:gz" if filename.endswith(".gz") else "r:"
        with tarfile.open(os.path.join(self.project_root, "tmp", filename), mode) as tar:
            tar.extractall(path=extract_to)

    def replace_mesh_paths(self, filename: str):
        # Read in the file
        with open(filename, 'r') as file:
            filedata = file.read()

        # replace mesh paths without file
        filedata = filedata.replace("filename=\"file://dataset_house_1/", "filename=\"file://resources/procthor_environments/dataset_house_1/dataset_house_1/")

        # Replace the target string
        filedata = filedata.replace('file://', f"{os.path.abspath(self.project_root)}/")
        # Write the file out again
        with open(filename, 'w') as file:
            file.write(filedata)

    def sample_environment(self, house_number: Optional[int] = None, keep_environment: bool = False):
        """
        Fetch and extract a random selection of environments packed in .tar.gz files from a URL.

        :param keep_environment: If True, the environments will be kept in the resources directory, otherwise they will be
        """
        if not keep_environment:
            output_dir = os.path.join(self.project_root, "tmp")
        else:
            output_dir = os.path.join(self.project_root, "resources/procthor_environments")
        if house_number is None:
            links = self.get_tarball_links()
            selected_link = random.choice(links)
            filename = os.path.basename(selected_link)
        else:
            filename = f"dataset_house_{house_number}.tar.gz"
        environment_name = os.path.splitext(os.path.splitext(filename)[0])[0]
        self.download_file(urljoin(self.base_url, filename), filename)
        self.extract_tar(filename, output_dir)
        os.remove(os.path.join(self.project_root, "tmp", filename))
        # self.replace_mesh_paths(os.path.join(self.project_root, "resources", "procthor_environments",
        #                                      environment_name, f"{environment_name}.urdf"))
        self.replace_mesh_paths(os.path.join(self.project_root, "resources", "procthor_environments",
                                             environment_name, f"{environment_name}.urdf"))
        return output_dir, environment_name

def main():
    procthor_interface = ProcTHORInterface(
        base_url="https://user.informatik.uni-bremen.de/~luc_kro/procthor_environments/"
    )
    house_number = 1
    resource_path, sampled_world = procthor_interface.sample_environment(
        house_number, keep_environment=True
    )

    world = URDFParser(
        os.path.join(
            resource_path, f"dataset_house_{house_number}", f"{sampled_world}.urdf"
        )
    ).parse()
    dressers = [b for b in world.bodies if "dresser" in b.name.name.lower()]

    semantic_world_database_uri = os.environ.get("SEMANTIC_WORLD_DATABASE_URI")

    # Create database engine and session
    engine = create_engine(f"mysql+pymysql://{semantic_world_database_uri}")
    session = Session(engine)

    for dresser in dressers:
        collision: Mesh = dresser.collision[0]
        dresser_id = collision.filename.split("/")[-1].split(".")[0].lower()

        dressers_on_bodies = (
            select(BodyDAO.id)
            .join(ContainerDAO, ContainerDAO.body_id == BodyDAO.id)  # body ↔ container
            .join(
                DresserDAO, DresserDAO.container_id == ContainerDAO.id
            )  # container ↔ dresser
            .join(
                PrefixedNameDAO, DresserDAO.name_id == PrefixedNameDAO.id
            )  # dresser ↔ its PrefixedName
            .where(
                PrefixedNameDAO.name == dresser_id,
                BodyDAO.worldmappingdao_bodies_id == WorldMappingDAO.id,
            )  # body belongs to this world
        )

        # outer query: worlds that have at least one such body
        query = select(WorldMappingDAO).where(exists(dressers_on_bodies))

        result = session.scalars(query).one()

        current_dresser_world: World = result.from_dao()

        parent_connection = dresser.parent_connection

        with world.modify_world():
            other_world_root = current_dresser_world.root
            parent_connection.child = other_world_root
            world.merge_world(current_dresser_world, parent_connection)
            world.remove_body(dresser)


    from semantic_world.adapters.viz_marker import VizMarkerPublisher
    import rclpy
    import time

    rclpy.init()

    node = rclpy.create_node("viz_marker")

    p = VizMarkerPublisher(world, node)
    time.sleep(100)
    p._stop_publishing()
    rclpy.shutdown()


if __name__ == "__main__":
    main()
