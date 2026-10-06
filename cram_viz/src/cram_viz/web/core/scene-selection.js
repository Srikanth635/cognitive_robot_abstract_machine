/* Keep every browser panel on the scene selected by the page URL. */
'use strict';

(function (global) {
  function nameFromSearch(search) {
    const name = new URLSearchParams(search || '').get('scene');
    return name && /^[\w-]+$/.test(name) ? name : null;
  }

  function apiUrl(url, search) {
    const name = nameFromSearch(search);
    if (!name) return url;
    const separator = url.indexOf('?') >= 0 ? '&' : '?';
    return url + separator + 'scene=' + encodeURIComponent(name);
  }

  global.SceneSelection = Object.freeze({
    nameFromSearch: nameFromSearch,
    apiUrl: apiUrl,
  });
})(window);
