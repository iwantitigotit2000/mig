// Build report/group_7_report.docx — the single-column (Word) version of the report; the submitted
// report/group_7_report.pdf was exported from this document after final editing in Google Docs.
// All result numbers are read from report/tables/numbers.tex (generated), so they stay consistent.
// Run: node report/make_docx.js
const fs = require("fs");
const path = require("path");
const sizeOf = (p) => { const b = fs.readFileSync(p); return { w: b.readUInt32BE(16), h: b.readUInt32BE(20) }; };
const {
  Document, Packer, Paragraph, TextRun, HeadingLevel, AlignmentType, Table, TableRow, TableCell, WidthType,
  ImageRun, BorderStyle, ShadingType, LevelFormat, PageBreak,
} = require("docx");

const ROOT = path.resolve(__dirname, "..");
const FIG = path.join(ROOT, "report/figures");

// ---- numbers from the generated macro file ------------------------------------------------
const N = {};
for (const line of fs.readFileSync(path.join(ROOT, "report/tables/numbers.tex"), "utf8").split("\n")) {
  const m = line.match(/^\\newcommand\{\\(\w+)\}\{(.*)\}$/);
  if (m) N[m[1]] = m[2].replace(/\$-\$/g, "−").replace(/\\%/g, "%").replace(/\\,/g, " ");
}
const mainRows = fs.readFileSync(path.join(ROOT, "report/tables/main_table.tex"), "utf8").split("\n")
  .filter((l) => l.includes("&")).map((l) => l.replace(/\\\\\s*$/, "").split("&").map((c) =>
    c.replace(/\\textbf\{|\}/g, "").replace(/\$\\uparrow\$/g, "↑").replace(/\$\\downarrow\$/g, "↓").replace(/\$-\$/g, "−").replace(/~~/g, "   ").trim()));
const ftRows = fs.readFileSync(path.join(ROOT, "report/tables/ft_table.tex"), "utf8").split("\n")
  .filter((l) => l.includes("&")).map((l) => l.replace(/\\\\\s*$/, "").split("&").map((c) =>
    c.replace(/\\textbf\{|\}/g, "").replace(/\$-\$/g, "−").trim()));

// ---- helpers -------------------------------------------------------------------------------
const P = (text, opts = {}) => new Paragraph({ spacing: { after: 120 }, alignment: AlignmentType.JUSTIFIED, ...opts,
  children: Array.isArray(text) ? text : [new TextRun({ text, size: 22 })] });
const B = (t) => new TextRun({ text: t, bold: true, size: 22 });
const T = (t) => new TextRun({ text: t, size: 22 });
const I = (t) => new TextRun({ text: t, italics: true, size: 22 });
const H1 = (t) => new Paragraph({ heading: HeadingLevel.HEADING_1, spacing: { before: 280, after: 120 }, children: [new TextRun({ text: t })] });
const H2 = (t) => new Paragraph({ heading: HeadingLevel.HEADING_2, spacing: { before: 200, after: 80 }, children: [new TextRun({ text: t })] });
const bullet = (runs) => new Paragraph({ numbering: { reference: "bullets", level: 0 }, spacing: { after: 60 },
  children: Array.isArray(runs) ? runs : [T(runs)] });
const caption = (t) => new Paragraph({ spacing: { after: 200 }, alignment: AlignmentType.CENTER,
  children: [new TextRun({ text: t, size: 18, italics: true, color: "444444" })] });
const figure = (file, widthCm, cap) => {
  const p = path.join(FIG, file); const { w, h } = sizeOf(p);
  const W = Math.round(widthCm * 37.8), Hh = Math.round(W * h / w);
  return [new Paragraph({ alignment: AlignmentType.CENTER, spacing: { before: 120, after: 60 },
    children: [new ImageRun({ type: "png", data: fs.readFileSync(p), transformation: { width: W, height: Hh } })] }), caption(cap)];
};
const cell = (t, { bold = false, shade = false, width, align = AlignmentType.LEFT } = {}) => new TableCell({
  width: { size: width, type: WidthType.DXA }, shading: shade ? { type: ShadingType.CLEAR, fill: "EDEDED", color: "auto" } : undefined,
  margins: { top: 40, bottom: 40, left: 80, right: 80 },
  children: [new Paragraph({ alignment: align, spacing: { after: 0 }, children: [new TextRun({ text: t, bold, size: 19 })] })] });
const table = (header, rows, widths) => new Table({ width: { size: widths.reduce((a, b) => a + b, 0), type: WidthType.DXA }, columnWidths: widths,
  rows: [new TableRow({ tableHeader: true, children: header.map((h, i) => cell(h, { bold: true, shade: true, width: widths[i] })) }),
    ...rows.map((r) => new TableRow({ children: r.map((c, i) => cell(c, { width: widths[i], align: i === 0 ? AlignmentType.LEFT : AlignmentType.RIGHT })) }))] });
const spacer = () => new Paragraph({ spacing: { after: 160 }, children: [] });

// ---- content -------------------------------------------------------------------------------
const children = [];
children.push(new Paragraph({ alignment: AlignmentType.CENTER, spacing: { after: 80 },
  children: [new TextRun({ text: "Layout-to-Image Generation with Multiple Objects:", bold: true, size: 34 })] }));
children.push(new Paragraph({ alignment: AlignmentType.CENTER, spacing: { after: 160 },
  children: [new TextRun({ text: "Comparing GLIGEN and MIGC on COCO-MIG", bold: true, size: 34 })] }));
children.push(new Paragraph({ alignment: AlignmentType.CENTER, spacing: { after: 40 },
  children: [new TextRun({ text: "CA6127 Transformer and Generative Adversarial Networks — Group Project (Technical Review)", size: 22 })] }));
children.push(new Paragraph({ alignment: AlignmentType.CENTER, spacing: { after: 320 },
  children: [new TextRun({ text: "Group 7: Zhichen Liu, Xinkuo Wu, Yanyu Wen", size: 22, italics: true })] }));

children.push(H1("1. Introduction"));
children.push(P("Text-to-image diffusion models such as Stable Diffusion (Rombach et al., 2022) follow a prompt well, but they handle two aspects of multi-object scenes poorly: placing each object at a requested position, and assigning each object its own attribute. For a prompt such as \"a red donut and a green donut and a blue donut\", the model often produces donuts with the colours swapped, or fewer donuts than requested."));
children.push(P("Layout-to-image (L2I) generation addresses this by adding a bounding box and a short phrase (\"a red donut\") for every object as extra conditioning. Because the boxes and phrases are known, the output can be checked automatically: whether a donut appears in the box, and whether it is red. With several objects per image the task is referred to as multi-instance generation (MIG)."));
children.push(P([T("This project compares two representative L2I methods that share the same backbone (Stable Diffusion 1.4; Rombach et al., 2022) but inject the layout in different ways: "), B("GLIGEN"), T(" (Li et al., 2023) and "), B("MIGC"), T(" (Zhou et al., 2024). Because the backbone is identical, differences between the two are more likely to come from the control mechanism than from the generator. The project addresses three questions:")]));
children.push(bullet("Which method follows the layout better, and where exactly does the other one fail (missing object, wrong place, or wrong colour)?"));
children.push(bullet("Which conditions make the task hard: many objects, overlapping boxes, same category in different colours, small objects?"));
children.push(bullet("Can the weaker method be improved by a cheap fine-tuning of its layout-related modules, and which hyper-parameters matter?"));
children.push(P("All images, evaluations and analyses reported here were produced by the group with the code in the submitted package. Results that could not be independently re-verified are marked as such in the text."));

children.push(H1("2. Methods"));
children.push(P("Both methods keep the SD 1.4 UNet frozen and add trainable modules. Tensor shapes were recorded with forward hooks on a 2-object example, so the descriptions below follow the released code rather than the papers alone. Figure 1 shows where each method plugs into the network."));
children.push(...figure("arch_gligen.png", 16, "Figure 1a. GLIGEN: a \"gated self-attention\" layer is inserted in every one of the 16 transformer blocks. Box + phrase are turned into 30 grounding tokens by position_net."));
children.push(...figure("arch_migc.png", 16, "Figure 1b. MIGC: cross-attention in 4 low-resolution layers is replaced by a per-instance controller (instance shading, Enhancement Attention, Layout Attention) whose outputs are blended by the Shading Aggregation Controller."));
children.push(H2("2.1 GLIGEN"));
children.push(P(`GLIGEN encodes each (box, phrase) pair into one "grounding token": the four box coordinates are Fourier-encoded (64 numbers), concatenated with the CLIP text feature of the phrase (768 numbers), and passed through a small MLP (832→512→512→768). There are always 30 token slots; unused slots are filled with a learned "null" token. This module, position_net, has only ${N.PosNetParamsM} M parameters.`));
children.push(P(`Inside every transformer block, after the normal self-attention, the image tokens are concatenated with the 30 grounding tokens and go through one more self-attention layer (the "fuser"). Its output is added back through a learned scalar gate: x ← x + tanh(α)·Attn([x; tokens]). The ${N.FuserCount} fusers hold ${N.FuserParamsM} M parameters (${N.FuserFrac} of the UNet). In the official checkpoint all gate values |tanh α| are below ${N.GateMaxAbs}. At inference the gate is only open for the first 30 % of the sampling steps (${N.GligenSteps} of 50); after that the model behaves like plain SD 1.4.`));
children.push(P("Note that nothing in this design ties the colour word of a phrase to its box: the phrase is encoded as a whole, and the global caption, which contains all colour words, is still processed by the frozen cross-attention."));
children.push(H2("2.2 MIGC"));
children.push(P(`MIGC only touches 4 layers: the mid block (8×8 tokens) and the three attention layers of up_blocks.1 (16×16 tokens), all with 1280 channels. In total ${N.MigcParamsM} M parameters are added (${N.MigcFrac} of the UNet). Inside these layers MIGC follows a divide-and-conquer scheme:`));
children.push(bullet([B("Instance shading. "), T("For every instance a separate masked cross-attention is computed between the image tokens and that instance's phrase, restricted to its box. Thus the phrase \"a red donut\" only interacts with the pixels inside its own box.")]));
children.push(bullet([B("Enhancement Attention (EA). "), T("Each instance attends once more to a 2-token context (the pooled phrase and a box token), which strengthens the response inside the box.")]));
children.push(bullet([B("Layout Attention (LA). "), T("A self-attention over the image tokens with a layout mask, producing a shared \"shading template\" (consistent background/lighting).")]));
children.push(bullet([B("Shading Aggregation Controller (SAC). "), T(`All candidates (one per instance + the normal caption attention + the LA template) are stacked and blended with learned per-pixel softmax weights. MIGC is active for the first ${N.MigcSteps} of 50 steps.`)]));
children.push(P("In summary, GLIGEN lets all grounding tokens interact inside one attention layer and relies on the frozen caption attention for appearance, whereas MIGC keeps the instances separate until a learned blending step merges them."));
children.push(table(["", "GLIGEN", "MIGC"], [
  ["How the layout enters", "30 tokens concatenated into self-attention", "one masked cross-attention per instance"],
  ["Layers touched", "all 16 blocks (64² down to 8²)", "4 layers (16² and 8²)"],
  ["Spatial constraint", "implicit (box is just part of a token)", "explicit (box mask + SAC blending)"],
  ["Instances isolated?", "no", "yes, until SAC"],
  ["Added parameters", `${N.GligenAddedM} M (${N.GligenAddedFrac})`, `${N.MigcParamsM} M (${N.MigcFrac})`],
  ["Control active for", `${N.GligenSteps}/50 steps (30 %)`, `${N.MigcSteps}/50 steps (50 %)`],
  ["Official sampler", "PLMS", "Euler"],
  ["Speed / memory (RTX 4070)", `${N.SecG} s per image, ${N.MemG} GB`, `${N.SecM} s per image, ${N.MemM} GB`],
], [3000, 3200, 3200]));
children.push(caption("Table 1. Implementation facts (from the official code and our probes)."));

children.push(H1("3. Experimental setup"));
children.push(H2("3.1 Dataset"));
children.push(P("We use COCO-MIG, the benchmark introduced with MIGC (Zhou et al., 2024). It contains 800 layouts taken from COCO val2014 boxes (Lin et al., 2014); each object is assigned one of seven colours (red, blue, green, yellow, black, white, brown) and images have 2 to 6 objects (levels L2–L6). The layouts were split by level into a test set of 100 layouts (20 per level), a validation set of 30 layouts used only for hyper-parameter choices, and a 10-layout pilot subset for debugging. The splits are fixed in the repository (with sha256 hashes). Every layout is generated with two random seeds, so each method produced 200 test images containing 800 target objects."));
children.push(P("From the inputs alone, each object was tagged with difficulty conditions: whether the same category appears in a different colour in the same image (57 % of objects), how much its box overlaps with another box (none / low / mid / high), and its size quartile. COCO-MIG contains no \"small\" objects in the COCO sense (the smallest box is 66 px wide at 512×512), so size is analysed by quartile."));
children.push(P("For fine-tuning, 3,000 images were taken from COCO train2014. train2014 and val2014 are disjoint image sets; in addition, every evaluation layout was searched against the training images for near-duplicates (0 of 130 matched at mean IoU ≥ 0.9). COCO image IDs could not be checked one by one because the benchmark file does not record them."));
children.push(H2("3.2 Generation settings"));
children.push(P(`All settings shared by the two methods were fixed: 512×512, 50 steps, CFG 7.5, the same negative prompt, the same initial noise for a given seed, fp16 autocast. Both wrappers reproduce the official inference scripts pixel-for-pixel. Two settings were left at each method's official value and therefore differ: the sampler (GLIGEN's code uses PLMS, MIGC's uses Euler) and how long the layout control is active (GLIGEN ${N.GligenSteps}/50 steps, MIGC ${N.MigcSteps}/50). Both differences are discussed in the limitations.`));
children.push(H2("3.3 Evaluation"));
children.push(P("The official MIG-Bench protocol (Zhou et al., 2024) was re-implemented: Grounding DINO (Liu et al., 2024) detects objects of the requested category; an object is \"in position\" if some detection overlaps the target box with IoU ≥ 0.5; SAM (Kirillov et al., 2023) segments the object inside the target box; the colour is \"correct\" if at least 20 % of the mask pixels fall in the HSV range of the requested colour. The main metric, referred to here as Attribute Success (identical to MIG-Bench instance success), requires both position and colour to be correct. Image Success means every object in the image succeeded."));
children.push(P("A few diagnostics were added under separate names so that they are not confused with the official metrics: Object Recall (the object was drawn somewhere — a one-to-one match exists), Localisation IoU (position quality ignoring colour), the failure type of each object (missing / mislocated / colour wrong), and Color Leakage Rate (the object's mask is dominated by a colour that was requested for another object in the same image). Image quality is measured with CLIP-IQA (Wang et al., 2023)."));
children.push(P("All confidence intervals are 95 % bootstrap intervals computed over layouts (both seeds and all objects of a layout are resampled together, 5,000 draws). Differences between methods are paired on the same (layout, seed, object)."));

children.push(H1("4. Results and analysis"));
children.push(H2("4.1 Overall comparison"));
children.push(P(`Table 2 and Figure 2 give the main result. ¹`));
children.push(table(["Metric", "GLIGEN", "MIGC", "MIGC − GLIGEN [95 % CI]"], mainRows, [3300, 1300, 1300, 3500]));
children.push(caption("Table 2. Test set (100 layouts × 2 seeds, 800 objects per method). Last column: paired difference with layout-bootstrap CI."));
children.push(...figure("test_paired.png", 16.5, `Figure 2. Left: paired difference MIGC − GLIGEN on every metric (orange = MIGC better). Right: per-layout Attribute Success; MIGC is better on ${N.LayoutsBetter} of ${N.LayoutsTotal} layouts.`));
children.push(P("Three observations follow from these results:"));
children.push(bullet([B("Both methods draw the objects. "), T(`Object Recall is ${N.RecallG} vs ${N.RecallM}, and the difference is not significant. MIGC places them slightly more accurately (Position Success ${N.PosDelta}, IoU ${N.IoUDelta}).`)]));
children.push(bullet([B("The gap is in colour. "), T(`Among objects that are in the right place, only ${N.ColPosG} of GLIGEN's have the right colour versus ${N.ColPosM} of MIGC's. The ${N.AttrGap} gap in Attribute Success is therefore accompanied mainly by a difference in colour pass rate, while position differences are small. This is not a causal decomposition: colour is only judged after the position test has been passed.`)]));
children.push(bullet([B("Image-level success amplifies instance-level differences. "), T(`With about ${N.InstPerImage} objects per image, GLIGEN produces a fully correct image in ${N.ImgG} of cases and MIGC in ${N.ImgM}. Image quality is essentially unchanged (CLIP-IQA difference ${N.IQADelta}), so MIGC's stronger control does not come at the expense of fidelity.`)]));
children.push(H2("4.2 Failure modes"));
children.push(...figure("test_failure_modes.png", 16.5, "Figure 3. Left: what happened to every target object. Right: for colour-wrong objects, the requested colour (rows) vs the colour that actually dominated the mask (columns)."));
children.push(P(`GLIGEN: ${N.ColWrongG} of all objects are colour-wrong, and ${N.LeakShareG} of those are leakage — the object took a colour that was requested for a different object. The confusion matrix in Figure 3 (left) is almost uniform: any colour can end up on any object. This is consistent with the architecture: the grounding tokens share one attention layer, and the colours are ultimately rendered by the frozen caption attention, which sees "a red donut and a green donut and …" with nothing tying "red" to a particular box. This is a plausible explanation rather than a result established by ablation.`));
children.push(P(`MIGC: ${N.ColWrongM} colour-wrong, ${N.LeakM} leakage, and ${N.ConfShare} of its colour errors (${N.ConfThree} of ${N.ConfTotal}) sit on just three pairs of neighbouring HSV colours: yellow→brown (${N.ConfYB}), brown→black (${N.ConfBrBl}) and yellow→black (${N.ConfYBl}). Visual inspection (Figure 4, right) shows tan-coloured cows judged "brown" and clearly brown vases judged "red" by the evaluator. Part of MIGC's remaining error is therefore a limitation of the HSV colour table rather than of the model.`));
children.push(P(`Missing and mislocated objects are rare for both methods (GLIGEN ${N.MissG} / ${N.MislocG}, MIGC ${N.MissM} / ${N.MislocM}). Several GLIGEN mislocations are objects that extend beyond their box; this may be related to the 70 % of steps during which its gate is closed, although this was not tested.`));
children.push(...figure("cases_A.png", 8.2, "Figure 4 (left). Category A: GLIGEN colour leakage on 6-object layouts where MIGC gets everything right."));
children.push(...figure("cases_E.png", 8.2, "Figure 4 (right). Category E: MIGC objects judged colour-wrong on yellow/brown boundaries, where the evaluator's colour table is arguably at fault. Boxes show the target layout in the target colour; the label is the evaluator's verdict."));
children.push(H2("4.3 Difficulty conditions"));
children.push(...figure("test_subgroups.png", 16.5, "Figure 5. Attribute Success by difficulty condition (95 % layout-bootstrap CIs)."));
children.push(bullet([B("Number of objects "), T(`is GLIGEN's main difficulty: ${N.LvlTwoG} with 2 objects down to ${N.LvlSixG} with 6, while MIGC stays flat (${N.LvlTwoM} → ${N.LvlSixM}). A plausible reading: more objects means more colour words competing inside one shared attention, whereas MIGC's instance branches do not see each other.`)]));
children.push(bullet([B("Box overlap "), T(`is MIGC's main difficulty: ${N.OvNoneM} with no overlap down to ${N.OvHighM} with high overlap, so the gap to GLIGEN shrinks from ${N.OvNoneDelta} to ${N.OvHighDelta} ${N.OvHighDeltaCI}. This contradicts our prior expectation that the SAC blending step would handle overlaps well. One possible explanation, not tested here, is that in an overlapping region two instance branches each paint their own colour and the weighted blend dilutes both, so one of them drops below the 20 % colour threshold. GLIGEN is flat here, possibly because it is already close to its floor.`)]));
children.push(bullet([B("Same category, different colour "), T(`changes neither method by more than ${N.SameCatMaxChange}. Colour errors are not specifically a same-category problem.`)]));
children.push(bullet([B("Small targets "), T(`(smallest quartile, box area below 5.1 % of the image) hurt GLIGEN substantially (${N.SizeQoneG} vs ${N.SizeQfourG} for the largest quartile) but not MIGC (${N.SizeQoneM} vs ${N.SizeQfourM}), even though MIGC only controls at 8×8 / 16×16 resolution where such a box covers only about 4–13 of the 256 tokens. This suggests that how the instances are kept apart matters more than the resolution of the control.`)]));
children.push(H2("4.4 Evaluator sensitivity"));
children.push(...figure("test_colour_threshold.png", 14, "Figure 6. Changing the evaluator's colour threshold (images, detections and masks fixed). The scores move; the gap does not."));
children.push(P(`The headline metric depends on a hand-picked threshold (20 % of mask pixels in the target colour range). Because the evaluator caches the colour ratio of every object, Attribute Success could be recomputed for thresholds from 0.05 to 0.5 without any GPU computation. Absolute scores fall almost linearly (GLIGEN ${N.ThrGHi} → ${N.ThrGLo}, MIGC ${N.ThrMHi} → ${N.ThrMLo}), but the paired gap stays between ${N.ThrGapLo} and ${N.ThrGapHi} and its intervals ${N.ThrGapAllPositive}. The ranking is therefore robust, but absolute numbers must be reported together with the threshold. The cache cannot be used to change the position threshold or the detector; that would require new GPU evaluation.`));
children.push(H2("4.5 Fine-tuning GLIGEN's layout modules"));
children.push(P(`GLIGEN's weak point is colour binding, which raises the question of whether its two layout modules can be adapted cheaply. Starting from the official checkpoint, we fine-tuned (G1) only position_net (${N.PosNetParamsM} M parameters, ${N.PosNetFrac} of the UNet) and (G2) position_net + the fusers (${N.GligenAddedM} M, ${N.GligenAddedFrac}) on 3,000 COCO train2014 images. The phrases only contain the category name (e.g. "dog"), so there is no colour supervision. Shared settings: 2,000 steps, effective batch 8, AdamW without weight decay, 100 warm-up steps then a constant learning rate, bf16, the official ε-MSE objective with 10 % grounding dropout, optimiser state reset; checkpoints at 500 and 2,000 steps; G2 was run with learning rates 5e-5 and 1e-5. All weight and gradient checks passed (frozen parameters unchanged, every trainable parameter updated).`));
children.push(...figure("train_curves.png", 16.5, `Figure 7. Training. (a) All runs see identical batches, so the loss curves overlap. (b) The same-batch difference shows G2 fits slightly better than G1. (c) Held-out loss on ${N.ValImages} COCO images saturates after about 1,250 steps. The absolute changes are below 0.3 % — expected with a frozen SD and no attribute supervision.`));
children.push(table(["Run", "Attr. Success", "Δ vs official [95 % CI]", "Colour | position", "Loc. IoU"], ftRows, [2200, 1500, 3100, 1600, 1200]));
children.push(caption("Table 3. Fine-tuned GLIGEN on the validation set (30 layouts × 2 seeds, grounding ratio 0.5). Exploratory — only 30 layouts."));
children.push(P([B("Learning rate. "), T(`With 1e-5, G2 is indistinguishable from the official model. With 5e-5 the validation point estimates are higher, but the direct contrast (G2 5e-5 vs 1e-5 at 500 steps: ${N.LrContrast}) includes zero. The data therefore only show that 1e-5 had no measurable effect, not that 5e-5 is better.`)]));
children.push(P([B("Checkpoint. "), T(`No consistent advantage of 2,000 over 500 steps was found (G1 even drops from ${N.GoneFiveHundred} to ${N.GoneTwoThousand}; all 2000 − 500 contrasts include zero). Held-out loss stops improving around 1,250 steps while the weights keep drifting (position_net ‖Δθ‖ reaches ${N.PosNetDrift}). Intermediate checkpoints were not evaluated on the task, so the exact onset of the plateau is unknown.`)]));
children.push(P([B("Grounding ratio. "), T("The group's Stage C-1 log reports that extending GLIGEN's gate from 30 % to 50 % of the steps gave a small but significant gain in localisation IoU and a smaller, non-significant change in Attribute Success. The raw outputs of that experiment are not in the code package, so no numbers from it are quoted here and it is treated as unverified. All fine-tuned runs use the ratio 0.5 that was chosen from that log.")]));
children.push(P([B("Where does the change appear? "), T(`The largest point-estimate changes are in colour-given-position (${N.ColPosValG} → ${N.ColPosValLo}–${N.ColPosValHi}) and in leakage (G2 5e-5 @500: ${N.LeakValG} → ${N.LeakValGtwo} ${N.LeakValCI}); localisation and object presence do not change consistently. Training only position_net gives about the same change as G2 (${N.FtGoneFive} vs ${N.FtGtwoFive}) with 1/${N.ParamRatio} of the trainable parameters and ${N.TrainMemGone} vs ${N.TrainMemGtwo} GB of GPU memory. The gate values tanh(α) barely move (< ${N.GateMaxChange}), which rules out a widening of the gate as the mechanism; whether the change sits in the fuser weights, in position_net, or in both cannot be determined from this evidence. With at most ${N.FtMaxGain} on validation and no colour supervision, this kind of fine-tuning is unlikely to close a ${N.AttrGap} gap. Our hypothesis, untested here, is that the binding problem lies in the frozen caption attention, which GLIGEN's design leaves untouched.`)]));

children.push(H1("5. Discussion"));
children.push(H2("5.1 Strengths and weaknesses of the two methods"));
children.push(P("GLIGEN is general (the same token mechanism can carry key-points or depth maps), it touches every resolution, and it has a clean training story (frozen SD, gates initialised at zero). However, its shared-attention injection has no notion of which instance owns which pixel: attributes leak, and performance drops with more objects and smaller boxes."));
children.push(P(`MIGC's per-instance branches give it strong attribute binding at low cost (${N.MigcFrac} extra parameters, 4 layers) and make it robust to the number of objects. However, its blending step degrades when boxes overlap, control only exists at 8×8 / 16×16 resolution, and because every instance gets its own branch, generation time grows with the number of objects (Figure 8), while GLIGEN's time is constant.`));
children.push(...figure("test_cost.png", 10, "Figure 8. Generation time vs number of objects (RTX 4070, fp16, 50 steps)."));
children.push(H2("5.2 Limitations of our study"));
children.push(bullet("No single-component ablation was run, so the architectural explanation of the gap is an interpretation consistent with the mechanism and the failure patterns, not an isolated causal result."));
children.push(bullet("100 test layouts × 2 seeds; confidence intervals are not adjusted for multiple comparisons; the validation set has only 30 layouts, so Section 4.5 is exploratory and several of its contrasts include zero."));
children.push(bullet(`Two generation settings differ between the methods (sampler PLMS vs Euler, control duration ${N.GligenSteps}/50 vs ${N.MigcSteps}/50 steps). Part of the gap could come from either; it cannot be bounded without re-running both methods under one sampler and one schedule.`));
children.push(bullet("Train/test isolation was verified at the source and layout level only; COCO image IDs are not available for the benchmark layouts."));
children.push(bullet("The HSV colour table misjudges yellow/brown/black boundaries, which penalises MIGC more than GLIGEN."));
children.push(bullet("The fine-tuned models were only evaluated on the validation set; the test-set evaluation of the chosen configurations (G1 and G2 5e-5) is still pending. The grounding-ratio experiment is only quoted qualitatively from the group log."));
children.push(bullet("The committed test images were generated on a local RTX 4070. The same experiment was also run on an A40 cluster and the numbers recorded in the group notes are very close, but the cluster files are not in the repository, so only the verifiable run is reported."));
children.push(H2("5.3 Future work"));
children.push(P("The evidence points to instance isolation as the key ingredient, but this has not been isolated experimentally. The most useful next step is therefore the missing ablation: (a) restrict each GLIGEN grounding token's attention to its own box and see whether that alone removes leakage; (b) fine-tune with colour-annotated phrases to separate \"cannot bind\" from \"never learned\"; (c) try an overlap-aware aggregation for MIGC (e.g. soft ownership or depth ordering) for the one condition where it degrades; (d) run both methods under one sampler (both code bases support DDIM) and test on a second benchmark such as HRS or T2I-CompBench."));

children.push(H1("6. Conclusion"));
children.push(P(`On COCO-MIG, with every shared setting fixed, GLIGEN and MIGC draw the requested objects equally often, but MIGC binds the right attribute to the right object about twice as reliably (${N.AttrM} vs ${N.AttrG}), and this ranking survives changes to the evaluator threshold. The most plausible explanation is architectural — shared grounding attention versus isolated instance branches. It is consistent with the opposite difficulty profiles observed here (GLIGEN fails with more objects, MIGC with overlapping boxes) and with the fact that cheaply fine-tuning GLIGEN's layout modules changes binding only marginally — but it has not been isolated by an ablation. In practice, an instance-isolating controller is preferable when per-object attributes must be correct; when boxes overlap heavily, neither method is yet reliable.`));

children.push(H1("References"));
const ref = (parts) => new Paragraph({ spacing: { after: 100 }, alignment: AlignmentType.LEFT, indent: { left: 720, hanging: 720 },
  children: parts.map(([t, it]) => new TextRun({ text: t, italics: !!it, size: 22 })) });
children.push(ref([["Kirillov, A., Mintun, E., Ravi, N., Mao, H., Rolland, C., Gustafson, L., Xiao, T., Whitehead, S., Berg, A. C., Lo, W.-Y., Dollár, P., & Girshick, R. (2023). Segment anything. In "], ["Proceedings of the IEEE/CVF International Conference on Computer Vision (ICCV)", 1], [" (pp. 4015–4026). IEEE."]]));
children.push(ref([["Li, Y., Liu, H., Wu, Q., Mu, F., Yang, J., Gao, J., Li, C., & Lee, Y. J. (2023). GLIGEN: Open-set grounded text-to-image generation. In "], ["Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR)", 1], [" (pp. 22511–22521). IEEE."]]));
children.push(ref([["Lin, T.-Y., Maire, M., Belongie, S., Hays, J., Perona, P., Ramanan, D., Dollár, P., & Zitnick, C. L. (2014). Microsoft COCO: Common objects in context. In D. Fleet, T. Pajdla, B. Schiele, & T. Tuytelaars (Eds.), "], ["Computer Vision – ECCV 2014", 1], [" (pp. 740–755). Springer."]]));
children.push(ref([["Liu, S., Zeng, Z., Ren, T., Li, F., Zhang, H., Yang, J., Jiang, Q., Li, C., Yang, J., Su, H., Zhu, J., & Zhang, L. (2024). Grounding DINO: Marrying DINO with grounded pre-training for open-set object detection. In "], ["Computer Vision – ECCV 2024", 1], [". Springer."]]));
children.push(ref([["Rombach, R., Blattmann, A., Lorenz, D., Esser, P., & Ommer, B. (2022). High-resolution image synthesis with latent diffusion models. In "], ["Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR)", 1], [" (pp. 10684–10695). IEEE."]]));
children.push(ref([["Wang, J., Chan, K. C. K., & Loy, C. C. (2023). Exploring CLIP for assessing the look and feel of images. "], ["Proceedings of the AAAI Conference on Artificial Intelligence, 37", 1], ["(2), 2555–2563."]]));
children.push(ref([["Zhou, D., Li, Y., Ma, F., Zhang, X., & Yang, Y. (2024). MIGC: Multi-instance generation controller for text-to-image synthesis. In "], ["Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR)", 1], [" (pp. 6818–6828). IEEE."]]));

children.push(spacer());
children.push(new Paragraph({ spacing: { after: 60 }, border: { top: { style: BorderStyle.SINGLE, size: 4, color: "999999" } }, children: [] }));
children.push(P([new TextRun({ text: "¹ Data note. ", size: 18, bold: true }), new TextRun({ size: 18, text: "The numbers in this report are computed from the test-set run committed in the repository; its metadata shows it was generated on an RTX 4070. The group notes also report a cluster (A40) run with the same seeds and settings (instance success 0.319 / 0.669 vs 0.321 / 0.665 here), but those files are not in the repository, so only the re-checked run is quoted. See outputs/final_analysis/PROVENANCE.md." })], { alignment: AlignmentType.LEFT }));
children.push(P([new TextRun({ text: "Reproducibility. ", size: 18, bold: true }), new TextRun({ size: 18, text: "From the repository root, bash scripts/stageD_analysis.sh checks the data splits and training/test isolation, recomputes all statistics, tables and figures from the committed per-object records, verifies every number used in the report, and rebuilds the PDF. The CPU pipeline was independently re-run on an Apple Silicon Mac (Python 3.12) on 18 October 2026 and all checks passed; that re-run did not repeat GPU generation, Grounding DINO/SAM/CLIP-IQA inference or fine-tuning." })], { alignment: AlignmentType.LEFT }));

const doc = new Document({
  creator: "Group 7", title: "GLIGEN vs MIGC on COCO-MIG",
  styles: { default: { document: { run: { font: "Calibri", size: 22 } } },
    paragraphStyles: [
      { id: "Heading1", name: "Heading 1", basedOn: "Normal", next: "Normal", quickFormat: true, run: { size: 30, bold: true, color: "1F3864" } },
      { id: "Heading2", name: "Heading 2", basedOn: "Normal", next: "Normal", quickFormat: true, run: { size: 25, bold: true, color: "2E5496" } }] },
  numbering: { config: [{ reference: "bullets", levels: [{ level: 0, format: LevelFormat.BULLET, text: "•", alignment: AlignmentType.LEFT,
    style: { paragraph: { indent: { left: 540, hanging: 300 } } } }] }] },
  sections: [{ properties: { page: { margin: { top: 1300, bottom: 1300, left: 1400, right: 1400 } } }, children }],
});
const out = path.join(ROOT, "report/group_7_report.docx");
Packer.toBuffer(doc).then((b) => { fs.writeFileSync(out, b); console.log("wrote", out); });
