// Building elements without writing markup: a tag, its properties, and what is inside.

const SVG_NAMESPACE = 'http://www.w3.org/2000/svg';

function build(node, properties, children) {
  Object.entries(properties).forEach(function ([key, value]) {
    if (value === null || value === undefined || value === false) return;
    if (key === 'class') node.setAttribute('class', value);
    else if (key.startsWith('on')) node.addEventListener(key.slice(2), value);
    else node.setAttribute(key, value === true ? '' : String(value));
  });
  node.append(...children.flat(Infinity).filter(function (child) {
    return child !== null && child !== undefined && child !== false;
  }));
  return node;
}

export function el(tag, properties = {}, ...children) {
  return build(document.createElement(tag), properties, children);
}

export function svg(tag, properties = {}, ...children) {
  return build(document.createElementNS(SVG_NAMESPACE, tag), properties, children);
}
