import { app } from "../../scripts/app.js";
import { api } from "../../scripts/api.js";

/**
 * Refresh a dropdown from its live source.
 *
 * Both of these lists live outside ComfyUI and can change while it is running:
 * the models installed in LM Studio, and the presets saved there. ComfyUI asks
 * for a node's option list once, at startup, so a refresh button is the
 * difference between restarting ComfyUI and clicking once.
 */
const TARGETS = {
  LMStudioLoadModel: {
    label: "刷新模型",
    widget: "model",
    route: (node) => {
      const base =
        node.widgets?.find((w) => w.name === "api_base_url")?.value ||
        "http://127.0.0.1:1234";
      return `/lmstudio_loader/models?base=${encodeURIComponent(base)}`;
    },
    pick: (data) => data?.models,
    empty: "LM Studio 未运行",
  },
  LLMChat: {
    label: "刷新预设",
    widget: "system_prompt",
    route: () => "/qwen_pe/presets",
    pick: (data) => data?.presets,
    empty: "读不到预设",
  },
};

/** Push a new option list into a ComfyUI combo widget, whatever its shape. */
function setComboOptions(widget, values) {
  if (!widget || !values?.length) return false;
  const previous = widget.value;

  if (Array.isArray(widget.options)) {
    widget.options = values;
  } else if (widget.options && Array.isArray(widget.options.values)) {
    widget.options.values = values;
  } else {
    return false;
  }

  // Keep the current pick when it survived the refresh.
  widget.value = values.includes(previous) ? previous : values[0];
  widget.callback?.(widget.value);
  return true;
}

async function refresh(node, button, target) {
  const widget = node.widgets?.find((w) => w.name === target.widget);
  if (!widget) return;

  const label = button.name;
  button.name = "刷新中…";

  try {
    const res = await api.fetchApi(target.route(node));
    const data = await res.json();
    const values = target.pick(data) ?? [];

    if (!values.length) {
      button.name = `刷新失败：${(data?.error || target.empty).slice(0, 40)}`;
    } else if (setComboOptions(widget, values)) {
      button.name = `${label}（${values.length} 个）`;
    } else {
      button.name = "刷新失败：控件形状不认识";
    }
  } catch (err) {
    console.error(`[QwenPE] ${label} 失败:`, err);
    button.name = "刷新失败，看控制台";
  }

  setTimeout(() => {
    if (button.name !== label) button.name = label;
  }, 4000);
  node.setSize(node.computeSize());
  app.graph?.setDirtyCanvas(true, true);
}

app.registerExtension({
  name: "QwenPE.RefreshLists",

  async beforeRegisterNodeDef(nodeType, nodeData) {
    const target = TARGETS[nodeData.name];
    if (!target) return;

    const onNodeCreated = nodeType.prototype.onNodeCreated;
    nodeType.prototype.onNodeCreated = function () {
      onNodeCreated?.apply(this, arguments);
      const node = this;
      const button = node.addWidget("button", target.label, null, () =>
        refresh(node, button, target)
      );
      button.serialize = false;
    };
  },
});
