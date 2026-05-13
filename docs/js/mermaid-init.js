// Initialise Mermaid for mkdocs rendering.
//
// Mermaid is loaded via the CDN script tag declared in
// mkdocs.yml's extra_javascript list. We start it once on page
// load with the theme matching the site's dark / light mode.

document.addEventListener("DOMContentLoaded", function () {
    if (typeof mermaid === "undefined") {
        return;
    }
    const scheme = document.body.getAttribute("data-md-color-scheme");
    const theme = scheme === "slate" ? "dark" : "default";
    mermaid.initialize({
        startOnLoad: true,
        theme: theme,
        flowchart: {
            useMaxWidth: true,
            htmlLabels: true,
        },
    });
});
