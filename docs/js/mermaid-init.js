// Initialise Mermaid for mkdocs rendering.
//
// pymdownx.superfences with `fence_code_format` emits
// `<pre class="mermaid"><code>...</code></pre>`.  Mermaid's
// auto-init looks for `.mermaid` containers and reads their
// text content directly, so the inner `<code>` wrapper trips
// it up.  We pre-process: for every `pre.mermaid`, replace it
// with a `<div class="mermaid">` carrying the raw graph source
// (read from the wrapped `<code>` element's text content),
// then call `mermaid.run()` to render.

document.addEventListener("DOMContentLoaded", function () {
    if (typeof mermaid === "undefined") {
        return;
    }
    document.querySelectorAll("pre.mermaid").forEach(function (pre) {
        const code = pre.querySelector("code");
        const source = (code ? code.textContent : pre.textContent) || "";
        const div = document.createElement("div");
        div.className = "mermaid";
        div.textContent = source.trim();
        pre.parentNode.replaceChild(div, pre);
    });
    const scheme = document.body.getAttribute("data-md-color-scheme");
    const theme = scheme === "slate" ? "dark" : "default";
    mermaid.initialize({
        startOnLoad: false,
        theme: theme,
        flowchart: {
            useMaxWidth: true,
            htmlLabels: true,
        },
    });
    mermaid.run({
        querySelector: "div.mermaid",
    });
});
