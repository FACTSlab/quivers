// In-container probe for QIEC entry points of a generated JavaScript
// program. Reads /io/source.js and /io/calls.json, a list of
// [entry, arguments] pairs, calls each `qiec_<entry>` with its arguments
// followed by the empty static, attachment, handler, and operation
// tables, and writes the results to /io/result.json.
const fs = require("fs");
const vm = require("vm");
const asValue = (value) => Array.isArray(value) ? Object.freeze(value.map(asValue)) : value;
const context = { console: console, Math: Math, Object: Object, Array: Array, JSON: JSON, Infinity: Infinity };
vm.createContext(context);
vm.runInContext(fs.readFileSync("/io/source.js", "utf8"), context);
const calls = JSON.parse(fs.readFileSync("/io/calls.json", "utf8"));
const results = calls.map(([entry, args]) => context["qiec_" + entry](...args.map(asValue), [], {}, {}, {}));
fs.writeFileSync("/io/result.json", JSON.stringify(results));
