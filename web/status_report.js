import { app } from "../../scripts/app.js";

/**
 * Nodes whose text output is meant for a person, not for the graph.
 *
 * A node's string output is invisible unless something consumes it, which makes
 * every diagnostic message these nodes produce ("all parameters applied", "the
 * output was truncated", "the size could not be resolved") easy to miss. Each of
 * them returns the text as `ui.text` as well, so it can be drawn here without
 * wiring anything up.
 *
 * The box does not exist until there is something to put in it. An empty panel
 * announcing that results will appear later is pure clutter on a node that has
 * not run yet — it costs vertical space on every workflow ever opened, to say
 * nothing.
 */
const REPORT_NODES = new Set([
  "LMStudioLoadModel",
  "LLMChat",
  "PEHintResolve",
]);

app.registerExtension({
  name: "QwenPE.StatusReport",

  async beforeRegisterNodeDef(nodeType, nodeData) {
    if (!REPORT_NODES.has(nodeData.name)) return;

    const onNodeCreated = nodeType.prototype.onNodeCreated;
    nodeType.prototype.onNodeCreated = function () {
      onNodeCreated?.apply(this, arguments);
      const node = this;

      const pre = document.createElement("pre");
      pre.style.cssText =
        "margin:0;padding:6px 8px;font-size:11px;line-height:1.45;color:#cfd8dc;" +
        "background:#161616;border:1px solid #2b2b2b;border-radius:4px;" +
        "white-space:pre-wrap;word-break:break-word;max-height:190px;overflow:auto;" +
        "display:none;";

      const widget = node.addDOMWidget("mil_report", "div", pre, {
        serialize: false,
        hideOnZoom: false,
        getValue: () => "",
        setValue: () => {},
      });
      widget.serialize = false;
      if (widget.options) widget.options.serialize = false;

      // -4 is the established "takes no space" size for a DOM widget here.
      const width = () => node.size?.[0] ?? 340;
      widget.computeSize = (w) =>
        node._milHasReport ? [w ?? width(), Math.max(56, pre.offsetHeight || 56)] : [w ?? width(), -4];

      node._milShowReport = (text) => {
        if (!text) return;
        node._milHasReport = true;
        pre.textContent = text;
        pre.style.display = "block";
        node.setSize(node.computeSize());
        app.graph?.setDirtyCanvas(true, true);
      };

      node.setSize([Math.max(340, width()), node.computeSize()[1]]);
    };

    const onExecuted = nodeType.prototype.onExecuted;
    nodeType.prototype.onExecuted = function (message) {
      onExecuted?.apply(this, arguments);
      const raw = message?.text;
      const text = Array.isArray(raw) ? raw[0] : raw;
      if (typeof text === "string" && text) this._milShowReport?.(text);
    };
  },
});
