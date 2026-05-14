// pymdownx.tabbed with `alternate_style: true` emits a
// `<div class="tabbed-set tabbed-alternate">` containing radio
// inputs, labels, and `<div class="tabbed-block">` panels.
// mkdocs-material ships JS that listens for label clicks and
// toggles an `is-active` class on the matching panel; the
// `terminal` theme does not.  This script wires the same
// click-to-activate behaviour so the tabs work in any theme.

document.addEventListener("DOMContentLoaded", function () {
    document.querySelectorAll(".tabbed-set.tabbed-alternate").forEach(
        function (tabset) {
            const inputs = tabset.querySelectorAll(
                "input[type=\"radio\"]",
            );
            const labels = tabset.querySelectorAll(
                ".tabbed-labels > label",
            );
            const blocks = tabset.querySelectorAll(
                ".tabbed-content > .tabbed-block",
            );

            function activate(index) {
                inputs.forEach(function (inp, i) {
                    inp.checked = i === index;
                });
                labels.forEach(function (lab, i) {
                    lab.classList.toggle("is-active", i === index);
                });
                blocks.forEach(function (blk, i) {
                    blk.classList.toggle("is-active", i === index);
                });
            }

            // Initial active state: the input with `checked`
            // attribute, or fall back to the first one.
            let initial = 0;
            inputs.forEach(function (inp, i) {
                if (inp.hasAttribute("checked")) {
                    initial = i;
                }
            });
            activate(initial);

            labels.forEach(function (label, i) {
                label.addEventListener("click", function (e) {
                    e.preventDefault();
                    activate(i);
                });
            });
        },
    );
});
