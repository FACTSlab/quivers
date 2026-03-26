// disable highlight.js auto-highlighting — we use pygments instead
document.addEventListener("DOMContentLoaded", function() {
    // remove all hljs classes that highlight.js may have added
    document.querySelectorAll("pre code.hljs").forEach(function(el) {
        el.className = el.className.replace(/\bhljs\b/g, "").trim();
    });
});

// prevent highlight.js from initializing in the first place
if (typeof hljs !== "undefined") {
    hljs.initHighlightingOnLoad = function() {};
    hljs.highlightBlock = function() {};
    hljs.highlightElement = function() {};
}
