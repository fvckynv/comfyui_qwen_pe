import { app } from "../../scripts/app.js";
import { api } from "../../scripts/api.js";

const NODE_NAME = "MultiImageLoader";
const MAX_IMAGES = 16;
const FIXED_OUTPUTS = 4; // count, image_batch, images_list, images_pipe

/* ------------------------------------------------------------------ styles */

const STYLE_ID = "mil-style";
function injectStyle() {
  if (document.getElementById(STYLE_ID)) return;
  const style = document.createElement("style");
  style.id = STYLE_ID;
  style.textContent = `
    .mil-root { width:100%; box-sizing:border-box; font-family:sans-serif; font-size:12px;
                padding:6px; border-radius:6px; min-height:150px; }
    .mil-root.dragover { outline:2px dashed #8ab4f8; outline-offset:-4px;
                         background:rgba(138,180,248,.07); }
    .mil-drop { border:1px dashed #666; border-radius:6px; padding:8px 6px; text-align:center;
                color:#aaa; cursor:pointer; margin-bottom:6px; user-select:none; }
    .mil-root.dragover .mil-drop { border-color:#8ab4f8; color:#8ab4f8; }
    .mil-bar { display:flex; justify-content:space-between; align-items:center; color:#999;
               margin-bottom:4px; font-size:11px; }
    .mil-clear { cursor:pointer; color:#e07b7b; }
    .mil-clear:hover { text-decoration:underline; }
    .mil-grid { display:grid; grid-template-columns:repeat(auto-fill, minmax(64px,1fr)); gap:5px;
                max-height:190px; overflow-y:auto; }
    .mil-cell { position:relative; padding-top:100%; border-radius:4px; overflow:hidden;
                background:#1e1e1e; border:1px solid #333; cursor:grab; }
    .mil-cell img { position:absolute; inset:0; width:100%; height:100%; object-fit:cover;
                    pointer-events:none; }
    .mil-cell.drag-src { opacity:.35; }
    .mil-cell.drop-dst { border-color:#8ab4f8; box-shadow:0 0 0 2px rgba(138,180,248,.5) inset; }
    .mil-idx { position:absolute; top:2px; left:2px; min-width:15px; height:15px; line-height:15px;
               padding:0 3px; border-radius:8px; background:rgba(0,0,0,.72); color:#8ab4f8;
               font-size:10px; text-align:center; pointer-events:none; }
    .mil-x { position:absolute; top:2px; right:2px; width:16px; height:16px; line-height:15px;
             text-align:center; border-radius:50%; background:rgba(0,0,0,.72); color:#fff;
             font-size:11px; cursor:pointer; }
    .mil-x:hover { background:#c0392b; }
    .mil-mv { position:absolute; bottom:2px; width:16px; height:15px; line-height:14px;
              text-align:center; border-radius:3px; background:rgba(0,0,0,.72); color:#ddd;
              font-size:10px; cursor:pointer; visibility:hidden; }
    .mil-cell:hover .mil-mv { visibility:visible; }
    .mil-mv:hover { background:#8ab4f8; color:#111; }
    .mil-mv.l { left:2px; }
    .mil-mv.r { right:2px; }
    .mil-tip { color:#777; font-size:10px; margin-top:4px; }
  `;
  document.head.appendChild(style);
}

/* ------------------------------------------------------------------- state */

const filesWidget = (node) => node.widgets?.find((w) => w.name === "files");

function readFiles(node) {
  try {
    const parsed = JSON.parse(filesWidget(node)?.value ?? "[]");
    return Array.isArray(parsed) ? parsed.filter((x) => typeof x === "string" && x) : [];
  } catch {
    return [];
  }
}

function writeFiles(node, list) {
  const w = filesWidget(node);
  if (w) w.value = JSON.stringify(list);
}

function viewUrl(name) {
  const parts = name.split("/");
  const file = parts.pop();
  const subfolder = parts.join("/");
  return api.apiURL(
    `/view?filename=${encodeURIComponent(file)}&type=input&subfolder=${encodeURIComponent(subfolder)}`
  );
}

/**
 * Trim/extend the per-image sockets at the tail so the node exposes exactly one
 * socket per uploaded image. The four fixed outputs come first, so every change
 * happens after them and existing links are never disturbed.
 */
function syncOutputs(node) {
  const n = Math.min(MAX_IMAGES, readFiles(node).length);
  const want = FIXED_OUTPUTS + n;

  for (let i = 0; i < 64 && (node.outputs?.length ?? 0) > want; i++) {
    node.removeOutput(node.outputs.length - 1);
  }
  for (let i = 0; i < 64 && (node.outputs?.length ?? 0) < want; i++) {
    node.addOutput(`image_${node.outputs.length - FIXED_OUTPUTS + 1}`, "IMAGE");
  }
  node.setSize(node.computeSize());
  app.graph?.setDirtyCanvas(true, true);
}

/* ------------------------------------------------------------------- upload */

async function uploadOne(file) {
  const body = new FormData();
  body.append("image", file);
  body.append("overwrite", "true");
  body.append("type", "input");

  const res = await api.fetchApi("/upload/image", { method: "POST", body });
  if (res.status !== 200) throw new Error(`HTTP ${res.status}`);
  const data = await res.json();
  return data.subfolder ? `${data.subfolder}/${data.name}` : data.name;
}

const isFileDrag = (event) =>
  Array.from(event?.dataTransfer?.types ?? []).includes("Files");

/* ------------------------------------------------------------------ gallery */

function buildGallery(node) {
  injectStyle();

  const root = document.createElement("div");
  root.className = "mil-root";

  const drop = document.createElement("div");
  drop.className = "mil-drop";
  drop.textContent = "拖拽图片到节点任意位置，或点击选择（可多选）";

  const bar = document.createElement("div");
  bar.className = "mil-bar";
  const info = document.createElement("span");
  const clear = document.createElement("span");
  clear.className = "mil-clear";
  clear.textContent = "清空";
  bar.append(info, clear);

  const grid = document.createElement("div");
  grid.className = "mil-grid";

  const tip = document.createElement("div");
  tip.className = "mil-tip";
  tip.textContent = "拖缩略图可排序 · ◀ ▶ 微调 · 角标 = <imageN> 编号";

  root.append(drop, bar, grid, tip);

  const picker = document.createElement("input");
  picker.type = "file";
  picker.accept = "image/*";
  picker.multiple = true;
  picker.style.display = "none";
  picker.addEventListener("change", async () => {
    await addFiles(picker.files);
    picker.value = "";
  });
  root.appendChild(picker);

  let dragSrc = -1; // thumbnail being reordered
  let fileDepth = 0; // dragenter/dragleave pairing for file drags

  const move = (from, to) => {
    const list = readFiles(node);
    if (from === to || from < 0 || to < 0 || from >= list.length || to >= list.length) return;
    const next = [...list];
    const [item] = next.splice(from, 1);
    next.splice(to, 0, item);
    writeFiles(node, next);
    render();
  };

  const render = () => {
    const list = readFiles(node);
    info.textContent = `已加载 ${list.length} / ${MAX_IMAGES}`;
    grid.replaceChildren();

    list.forEach((name, index) => {
      const cell = document.createElement("div");
      cell.className = "mil-cell";
      cell.title = `${name}\n（对应 <image${index + 1}>）`;
      cell.draggable = true;

      const img = document.createElement("img");
      img.src = viewUrl(name);
      img.loading = "lazy";
      img.draggable = false; // let the cell be the drag source, not the bitmap

      const badge = document.createElement("span");
      badge.className = "mil-idx";
      badge.textContent = String(index + 1);

      const x = document.createElement("span");
      x.className = "mil-x";
      x.textContent = "×";
      x.title = "移除";
      x.addEventListener("click", (event) => {
        event.stopPropagation();
        writeFiles(node, readFiles(node).filter((_, i) => i !== index));
        render();
      });

      const left = document.createElement("span");
      left.className = "mil-mv l";
      left.textContent = "◀";
      left.title = "前移";
      left.addEventListener("click", (event) => {
        event.stopPropagation();
        move(index, index - 1);
      });

      const right = document.createElement("span");
      right.className = "mil-mv r";
      right.textContent = "▶";
      right.title = "后移";
      right.addEventListener("click", (event) => {
        event.stopPropagation();
        move(index, index + 1);
      });

      // --- 缩略图之间拖拽排序 ---
      cell.addEventListener("dragstart", (event) => {
        if (isFileDrag(event)) return; // 外部文件，交给 root 处理
        dragSrc = index;
        cell.classList.add("drag-src");
        event.dataTransfer.effectAllowed = "move";
        event.dataTransfer.setData("text/plain", String(index));
        event.stopPropagation();
      });
      cell.addEventListener("dragend", () => {
        dragSrc = -1;
        grid.querySelectorAll(".mil-cell").forEach((c) => {
          c.classList.remove("drag-src", "drop-dst");
        });
      });
      cell.addEventListener("dragover", (event) => {
        if (dragSrc < 0 || isFileDrag(event)) return; // 正在外部拖文件
        event.preventDefault();
        event.stopPropagation();
        cell.classList.add("drop-dst");
      });
      cell.addEventListener("dragleave", () => cell.classList.remove("drop-dst"));
      cell.addEventListener("drop", (event) => {
        if (dragSrc < 0 || isFileDrag(event)) return;
        event.preventDefault();
        event.stopPropagation();
        cell.classList.remove("drop-dst");
        move(dragSrc, index);
      });

      cell.append(img, badge, x, left, right);
      grid.appendChild(cell);
    });

    syncOutputs(node);
  };

  async function addFiles(fileList) {
    const incoming = [...(fileList ?? [])].filter((f) => f.type.startsWith("image/"));
    if (!incoming.length) return;
    drop.textContent = `上传中 0 / ${incoming.length} …`;

    const uploaded = [];
    for (let i = 0; i < incoming.length; i++) {
      try {
        uploaded.push(await uploadOne(incoming[i]));
      } catch (err) {
        console.error("[MultiImageLoader] 上传失败:", incoming[i]?.name, err);
      }
      drop.textContent = `上传中 ${i + 1} / ${incoming.length} …`;
    }

    drop.textContent = "拖拽图片到节点任意位置，或点击选择（可多选）";
    writeFiles(node, [...readFiles(node), ...uploaded].slice(0, MAX_IMAGES));
    render();
  }

  // --- 文件拖拽：整个 DOM 区域都是目标 ---
  root.addEventListener("dragenter", (event) => {
    if (!isFileDrag(event)) return;
    event.preventDefault();
    event.stopPropagation();
    fileDepth += 1;
    root.classList.add("dragover");
  });
  root.addEventListener("dragover", (event) => {
    if (!isFileDrag(event)) return;
    event.preventDefault();
    event.stopPropagation();
    if (event.dataTransfer) event.dataTransfer.dropEffect = "copy";
    root.classList.add("dragover");
  });
  root.addEventListener("dragleave", (event) => {
    if (!isFileDrag(event)) return;
    fileDepth -= 1;
    if (fileDepth <= 0) {
      fileDepth = 0;
      root.classList.remove("dragover");
    }
  });
  root.addEventListener("drop", async (event) => {
    if (!isFileDrag(event)) return;
    event.preventDefault();
    event.stopPropagation();
    fileDepth = 0;
    root.classList.remove("dragover");
    const dropped = event.dataTransfer?.files;
    if (dropped?.length) await addFiles(dropped);
  });

  drop.addEventListener("click", () => picker.click());
  clear.addEventListener("click", () => {
    writeFiles(node, []);
    render();
  });

  render();
  return { root, render };
}

/* ---------------------------------------------------------------- extension */

app.registerExtension({
  name: "MultiImageLoader.Gallery",

  async beforeRegisterNodeDef(nodeType, nodeData) {
    if (nodeData.name !== NODE_NAME) return;

    const onNodeCreated = nodeType.prototype.onNodeCreated;
    nodeType.prototype.onNodeCreated = function () {
      onNodeCreated?.apply(this, arguments);
      const node = this;

      const w = filesWidget(node);
      if (w) {
        w.type = "hidden";
        w.hidden = true;
        w.computeSize = () => [0, -4];
      }

      const { root, render } = buildGallery(node);
      const domWidget = node.addDOMWidget("mil_gallery", "div", root, {
        serialize: false,
        hideOnZoom: false,
        getValue: () => "",
        setValue: () => {},
      });
      domWidget.serialize = false;
      if (domWidget.options) domWidget.options.serialize = false;

      domWidget.computeSize = (width) => [
        width ?? node.size?.[0] ?? 320,
        Math.max(190, root.offsetHeight || 190),
      ];

      node._milRender = render;
      node.setSize([Math.max(320, node.size?.[0] ?? 320), node.computeSize()[1]]);
    };

    const onConfigure = nodeType.prototype.onConfigure;
    nodeType.prototype.onConfigure = function () {
      onConfigure?.apply(this, arguments);
      requestAnimationFrame(() => {
        try {
          this._milRender?.();
        } catch (err) {
          console.error("[MultiImageLoader] 恢复渲染失败:", err);
        }
        syncOutputs(this);
      });
    };
  },
});
