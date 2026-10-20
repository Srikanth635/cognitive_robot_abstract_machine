# Phase 7 deployment checkpoint

The fork's Actions service is enabled, and the first real publication has passed hosted admission and authenticated consumption. The policy is automated: no independent reviewer or approval environment is required. Concepts retain their draft lifecycle and do not acquire a human-review claim.

## Trusted identities

| Item | Identity |
| --- | --- |
| Checker and first source | `08b0d94cc33ed8b16cc6a1c2404874cb496805a4` |
| Runtime image | `ghcr.io/srikanth635/cram-knowledge-runtime@sha256:41944c40a8a456644cd6a29afc8d9256f7dec2f68cf69314a7d1996a5ae7de83` |
| First publication artifact | `12292221849` |
| Artifact archive digest | `sha256:e2d5e8d178491464a8e61d1efd47804d99a34830db198547c8b66b3f5b4b6ef2` |
| Bundle digest | `e4d878f2959b8216056152fad9b278dfca0596751205c06cc5003aeb812cc216` |
| Policy digest | `956465a19e76276849cf64cc1cf83a5cdf58d14bb604015e67c990c6935ab4d5` |

The owner-selected activation file lives on the fork's `main` branch at `repository_knowledge/policy/activation.env`. It is separate from proposed wiki/source changes. The publisher ref stays at the trusted checker while `cram_wiki` can advance independently. Main contains workflow registration and activation data; the implementation remains on `cram_wiki`.

## Hosted evidence

- [Runtime build and isolated import check](https://github.com/Srikanth635/cognitive_robot_abstract_machine/actions/runs/39482213639): passed.
- [Admission of the real source snapshot](https://github.com/Srikanth635/cognitive_robot_abstract_machine/actions/runs/39482597365): passed, with matching independent extraction fingerprints, no governance changes and no trust findings. The snapshot contains 3,184 source entries.
- [First publication](https://github.com/Srikanth635/cognitive_robot_abstract_machine/actions/runs/39482597606): passed and uploaded all eight source-backed wiki concepts with their cited source projection.
- [Authenticated consumption of the first artifact](https://github.com/Srikanth635/cognitive_robot_abstract_machine/actions/runs/39483469254): passed using the immutable artifact ID and exact source commit.

The final source/artifact pair and the subsequent historical-restoration evidence are recorded in the owner-managed [verification report on main](https://github.com/Srikanth635/cognitive_robot_abstract_machine/blob/main/repository_knowledge/docs/phase7-verification.md). That report can record completed deployments without changing the source snapshot being published.

## Operating contract

Use the commands in [phase7.md](phase7.md) with a clean checker checkout and its matching KRROOD source. Select the exact artifact and source commit when consuming. Restoring an older publication returns its complete bundle and cited source facts; it does not reset Git or overwrite the working tree.

Source-only fallback remains mandatory for a refused publication. An artifact does not authenticate prose semantics, static resolution does not establish runtime behavior, and a declared test reference does not prove coverage. Agent-client integration of verified publications is phase 8 work.

Local validation passed 211 package tests with one Windows-only skip under the 4 GiB memory cap. Hosted checks exercised the real repository through the pinned runtime; a hosted run of the entire robotics test suite was not required for these publication controls.
