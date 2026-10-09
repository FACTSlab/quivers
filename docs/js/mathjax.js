window.MathJax = {
  tex: {
    inlineMath: [["\\(", "\\)"]],
    displayMath: [["\\[", "\\]"]],
    processEscapes: true,
    processEnvironments: true,
    tags: "ams",
    useLabelIds: true,
    macros: {
      llbracket: "[\\![",
      rrbracket: "]\\!]",
      semb: ["\\llbracket #1 \\rrbracket", 1]
    }
  },
  options: {
    ignoreHtmlClass: ".*|",
    // Markdown headings are copied into Material's table of contents as plain
    // text, so include navigation links when a heading contains inline TeX.
    processHtmlClass: "arithmatex|md-nav__link|md-path__link|md-ellipsis"
  }
};

// Material's instant navigation replaces the article without reloading the
// page. Re-typeset the new article and reset equation numbering on each page.
if (typeof document$ !== "undefined") {
  document$.subscribe(function () {
    if (!window.MathJax || !MathJax.typesetPromise) {
      return;
    }
    MathJax.startup.output.clearCache();
    MathJax.typesetClear();
    MathJax.texReset();
    MathJax.typesetPromise();
  });
}
