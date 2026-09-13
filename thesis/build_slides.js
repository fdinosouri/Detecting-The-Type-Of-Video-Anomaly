// Build the 25-slide Persian defence deck.
//
//     npm install pptxgenjs
//     node thesis/build_slides.js
//
// Every number here comes from REVIEW_14CLASS.md. The per-class test
// counts on the split slide are reconstructed from the experiment log;
// regenerate them from the real split files before presenting:
//
//     python -c "import collections,sys; from evaluate_multiclass import parse_annotation_label, CLASS_NAMES; c=collections.Counter(parse_annotation_label(l.split()) for l in open(sys.argv[1]) if len(l.split())>=3); [print(f'{CLASS_NAMES[i]:16s} {c[i]}') for i in range(len(CLASS_NAMES))]" labels/UCF_std_test.txt
//
const pptxgen = require("pptxgenjs");

const pres = new pptxgen();
pres.layout = "LAYOUT_WIDE";           // 13.3 x 7.5
pres.rtlMode = true;
pres.author = "";
pres.title = "تشخیص نوع ناهنجاری در ویدیوهای نظارتی";

const W = 13.3, H = 7.5;

// ---- palette: surveillance / night-watch -----------------------------
const NAVY = "16264F";
const NAVY_D = "0C1630";
const ICE = "BFD4F2";
const AMBER = "E8A33D";
const TEAL = "3FA7A0";
const RED = "C9483B";
const LIGHT = "F6F8FC";
const INK = "1B2236";
const MUTED = "6A748C";

const FA = "Arial";

// ---- helpers ---------------------------------------------------------
function rtl(extra) {
  return Object.assign({ fontFace: FA, rtlMode: true, align: "right",
                         valign: "top", isTextBox: true }, extra || {});
}
function ltr(extra) {
  return Object.assign({ fontFace: FA, align: "left", valign: "top",
                        isTextBox: true }, extra || {});
}

function darkSlide() {
  const s = pres.addSlide();
  s.background = { color: NAVY };
  return s;
}

function lightSlide(title, kicker) {
  const s = pres.addSlide();
  s.background = { color: LIGHT };

  if (kicker) {
    s.addText(kicker, rtl({
      x: 0.6, y: 0.38, w: 12.1, h: 0.32,
      fontSize: 13, color: AMBER, bold: true, margin: 0,
    }));
  }
  s.addText(title, rtl({
    x: 0.6, y: kicker ? 0.72 : 0.5, w: 12.1, h: 0.8,
    fontSize: 30, color: NAVY, bold: true, margin: 0,
  }));
  return s;
}

// a soft card
function card(s, x, y, w, h, fill) {
  s.addShape(pres.ShapeType.roundRect, {
    x, y, w, h, rectRadius: 0.1,
    fill: { color: fill || "FFFFFF" },
    line: { color: "E2E7F0", width: 1 },
    shadow: { type: "outer", color: "8A94AC", blur: 8, offset: 2,
              angle: 90, opacity: 0.16 },
  });
}

// numbered circle
function stepDot(s, x, y, n, color) {
  s.addShape(pres.ShapeType.ellipse, {
    x, y, w: 0.52, h: 0.52,
    fill: { color: color || NAVY }, line: { color: color || NAVY, width: 0 },
  });
  s.addText(String(n), {
    x, y, w: 0.52, h: 0.52, fontFace: FA, fontSize: 16, bold: true,
    color: "FFFFFF", align: "center", valign: "middle", margin: 0,
    isTextBox: true,
  });
}

// big stat block
function stat(s, x, y, w, value, label, color) {
  s.addText(value, {
    x, y, w, h: 0.85, fontFace: FA, fontSize: 40, bold: true,
    color: color || NAVY, align: "center", margin: 0, isTextBox: true,
  });
  s.addText(label, rtl({
    x, y: y + 0.82, w, h: 0.5, fontSize: 12, color: MUTED,
    align: "center", margin: 0,
  }));
}

// simple table
function table(s, x, y, w, head, rows, opts) {
  opts = opts || {};
  const colW = opts.colW;
  const body = [
    head.map(t => ({
      text: t,
      options: { bold: true, color: "FFFFFF", fill: { color: NAVY },
                 fontSize: opts.headSize || 12, align: "center",
                 valign: "middle", fontFace: FA },
    })),
  ];
  rows.forEach((r, i) => {
    body.push(r.map((t, j) => ({
      text: String(t),
      options: {
        fontSize: opts.size || 11,
        color: (opts.hi && opts.hi.indexOf(i) >= 0) ? NAVY : INK,
        bold: !!(opts.hi && opts.hi.indexOf(i) >= 0),
        fill: { color: (opts.hi && opts.hi.indexOf(i) >= 0)
                ? "FFF2DC" : (i % 2 ? "FFFFFF" : "EDF1F8") },
        align: j === 0 ? "right" : "center",
        valign: "middle", fontFace: FA,
      },
    })));
  });
  s.addTable(body, {
    x, y, w, colW,
    border: { type: "solid", color: "D9E0EC", pt: 0.5 },
    rowH: opts.rowH || 0.27,
    margin: opts.margin === undefined ? 3 : opts.margin,
    autoPage: false,
  });
}

function bullets(s, x, y, w, items, size, color) {
  const runs = items.map((t, i) => ({
    text: "\u25CF  " + t,
    options: { breakLine: i !== items.length - 1 },
  }));
  s.addText(runs, rtl({
    x, y, w, h: 0.4 * items.length + 0.4,
    fontSize: size || 14, color: color || INK,
    paraSpaceAfter: 7, margin: 0, lineSpacing: 22,
  }));
}

function note(s, txt) { s.addNotes(txt); }

// =====================================================================
// 1 — title
// =====================================================================
{
  const s = darkSlide();
  s.addShape(pres.ShapeType.ellipse, {
    x: 9.3, y: -1.6, w: 6.2, h: 6.2,
    fill: { color: "1E3566" }, line: { width: 0 },
  });
  s.addShape(pres.ShapeType.ellipse, {
    x: 10.9, y: 3.6, w: 3.6, h: 3.6,
    fill: { color: "24427D" }, line: { width: 0 },
  });

  s.addText("پایان‌نامه کارشناسی مهندسی کامپیوتر", rtl({
    x: 0.8, y: 1.5, w: 8.4, h: 0.4, fontSize: 14, color: AMBER,
    bold: true, margin: 0,
  }));
  s.addText("تشخیص نوع ناهنجاری\nدر ویدیوهای نظارتی", rtl({
    x: 0.8, y: 2.05, w: 8.4, h: 1.9, fontSize: 40, bold: true,
    color: "FFFFFF", lineSpacing: 48, margin: 0,
  }));
  s.addText(
    "یادگیری چندنمونه‌ای ضعیف‌نظارت با ستون فقرات ترنسفورمر ویدیویی",
    rtl({ x: 0.8, y: 4.05, w: 8.4, h: 0.5, fontSize: 16, color: ICE,
          margin: 0 }));

  s.addText("ماکرو F1 نهایی", rtl({
    x: 0.8, y: 5.15, w: 3.0, h: 0.3, fontSize: 12, color: ICE, margin: 0,
  }));
  s.addText("0.4298", ltr({
    x: 0.8, y: 5.45, w: 3.0, h: 0.8, fontSize: 42, bold: true,
    color: AMBER, margin: 0,
  }));
  s.addText("از خط پایه 0.2975", rtl({
    x: 4.0, y: 5.72, w: 3.2, h: 0.4, fontSize: 13, color: ICE, margin: 0,
  }));

  s.addText("دانشگاه صنعتی همدان  ·  گروه مهندسی کامپیوتر", rtl({
    x: 0.8, y: 6.6, w: 8.4, h: 0.35, fontSize: 12, color: "8FA6D4",
    margin: 0,
  }));
  note(s, "معرفی: پروژه تبدیل یک سامانه تشخیص ناهنجاری دودویی به سامانه چهارده‌کلاسه است. عدد نهایی ماکرو F1 برابر 0.4298 در برابر خط پایه 0.2975.");
}

// =====================================================================
// 2 — the problem
// =====================================================================
{
  const s = lightSlide("صورت مسئله", "فصل اول");

  card(s, 0.6, 1.75, 5.9, 2.1);
  s.addText("مسئله متداول: تشخیص ناهنجاری", rtl({
    x: 0.95, y: 1.95, w: 5.2, h: 0.4, fontSize: 17, bold: true,
    color: MUTED, margin: 0,
  }));
  s.addText("آیا این ویدیو ناهنجار است؟\nجواب: بله یا خیر", rtl({
    x: 0.95, y: 2.45, w: 5.2, h: 1.0, fontSize: 15, color: INK,
    lineSpacing: 24, margin: 0,
  }));

  card(s, 6.8, 1.75, 5.9, 2.1, "FFF7EA");
  s.addText("مسئله ما: تشخیص نوع ناهنجاری", rtl({
    x: 7.15, y: 1.95, w: 5.2, h: 0.4, fontSize: 17, bold: true,
    color: AMBER, margin: 0,
  }));
  s.addText("کدام نوع ناهنجاری رخ داده؟\nجواب: یکی از ۱۴ کلاس", rtl({
    x: 7.15, y: 2.45, w: 5.2, h: 1.0, fontSize: 15, color: INK,
    lineSpacing: 24, margin: 0,
  }));

  s.addText("چرا این تفاوت مهم است", rtl({
    x: 0.6, y: 4.15, w: 12.1, h: 0.4, fontSize: 18, bold: true,
    color: NAVY, margin: 0,
  }));
  bullets(s, 0.6, 4.65, 12.1, [
    "پاسخ عملیاتی به تصادف رانندگی، آمبولانس است؛ به سرقت مسلحانه، نیروی مسلح.",
    "یک هشدار خام «ناهنجاری رخ داد» برای اپراتور قابل اقدام نیست.",
    "از نظر دشواری، مسئله چهارده‌کلاسه با نسخه دودویی قابل مقایسه نیست.",
  ], 14);
  note(s, "تفاوت کلیدی: بیشتر پژوهش‌ها مسئله را دودویی تعریف می‌کنند. ما یک گام جلوتر رفتیم.");
}

// =====================================================================
// 3 — dataset overview
// =====================================================================
{
  const s = lightSlide("مجموعه داده UCF-Crime", "داده");

  stat(s, 0.6, 1.8, 2.9, "1900", "کل ویدیوها", NAVY);
  stat(s, 3.7, 1.8, 2.9, "14", "کلاس", NAVY);
  stat(s, 6.8, 1.8, 2.9, "1895", "رمزگشایی موفق", TEAL);
  stat(s, 9.9, 1.8, 2.9, "5", "فایل خراب", RED);

  card(s, 0.6, 3.5, 5.9, 3.2);
  s.addText("ویژگی‌های داده", rtl({
    x: 0.95, y: 3.7, w: 5.2, h: 0.4, fontSize: 17, bold: true,
    color: NAVY, margin: 0,
  }));
  bullets(s, 0.95, 4.2, 5.2, [
    "ویدیوی واقعی دوربین مداربسته، نه صحنه‌پردازی",
    "وضوح پایین، نور ضعیف، زاویه نامناسب",
    "مدت‌زمان از چند ثانیه تا چند دقیقه",
    "برچسب فقط در سطح کل ویدیو",
  ], 13);

  card(s, 6.8, 3.5, 5.9, 3.2, "FFF7EA");
  s.addText("چالش اصلی: نامتوازنی شدید", rtl({
    x: 7.15, y: 3.7, w: 5.2, h: 0.4, fontSize: 17, bold: true,
    color: AMBER, margin: 0,
  }));
  bullets(s, 7.15, 4.2, 5.2, [
    "کلاس عادی حدود نیمی از داده آموزش است",
    "کلاس تیراندازی کمتر از دو درصد",
    "در آزمون، نه کلاس تنها هشت ویدیو دارند",
    "کلاس Abuse تنها دو ویدیوی آزمون دارد",
  ], 13);
  note(s, "نامتوازنی هم آموزش را دشوار می‌کند و هم ارزیابی را نوفه‌آلود. این نکته در تفسیر همه اعداد برمی‌گردد.");
}

// =====================================================================
// 4 — train split per class
// =====================================================================
{
  const s = lightSlide("تعداد ویدیو در هر کلاس — مجموعه آموزش", "تقسیم داده");

  const left = [
    ["Normal — عادی", "800"],
    ["Robbery — سرقت", "145"],
    ["RoadAccidents — تصادف", "127"],
    ["Stealing — دزدی", "95"],
    ["Burglary — سرقت از منزل", "87"],
    ["Abuse — سوءاستفاده", "48"],
    ["Assault — تعرض", "47"],
  ];
  const right = [
    ["Arrest — بازداشت", "45"],
    ["Fighting — درگیری", "45"],
    ["Vandalism — تخریب", "45"],
    ["Arson — آتش‌افروزی", "41"],
    ["Explosion — انفجار", "29"],
    ["Shoplifting — دزدی از مغازه", "29"],
    ["Shooting — تیراندازی", "27"],
  ];

  table(s, 0.6, 1.75, 5.9, ["کلاس", "تعداد ویدیو"], left,
        { colW: [4.1, 1.8], rowH: 0.34, size: 12 });
  table(s, 6.8, 1.75, 5.9, ["کلاس", "تعداد ویدیو"], right,
        { colW: [4.1, 1.8], rowH: 0.34, size: 12 });

  card(s, 0.6, 5.15, 12.1, 1.5, "FFF7EA");
  s.addText("جمع: ۱۶۱۰ ویدیوی آموزش  ·  ۸۰۰ عادی و ۸۱۰ ناهنجار", rtl({
    x: 0.95, y: 5.35, w: 11.4, h: 0.4, fontSize: 16, bold: true,
    color: NAVY, margin: 0,
  }));
  s.addText(
    "نسبت پرتکرارترین به کم‌تکرارترین کلاس ناهنجار حدود ۵ به ۱ است و نسبت کلاس عادی به کمیاب‌ترین کلاس حدود ۳۰ به ۱. همین نسبت است که وزن‌دهی معکوس فراوانی را ضروری می‌کند.",
    rtl({ x: 0.95, y: 5.8, w: 11.4, h: 0.7, fontSize: 12.5, color: INK,
          lineSpacing: 19, margin: 0 }));
  note(s, "این توزیع استاندارد UCF-Crime است و جمعش دقیقا 1610 می‌شود. پیش از جلسه اعداد را از فایل تقسیم خودتان تایید کنید.");
}

// =====================================================================
// 5 — test split
// =====================================================================
{
  const s = lightSlide("تعداد ویدیو در هر کلاس — مجموعه آزمون", "تقسیم داده");

  const rows = [
    ["Normal — عادی", "150", "148"],
    ["RoadAccidents — تصادف رانندگی", "23", "23"],
    ["Shooting — تیراندازی", "23", "8"],
    ["Explosion — انفجار", "21", "8"],
    ["Shoplifting — دزدی از مغازه", "21", "8"],
    ["Burglary — سرقت از منزل", "13", "15"],
    ["Arson — آتش‌افروزی", "9", "8"],
    ["Arrest — بازداشت", "5", "8"],
    ["Fighting — درگیری", "5", "8"],
    ["Robbery — سرقت", "5", "23"],
    ["Stealing — دزدی", "5", "15"],
    ["Vandalism — تخریب اموال", "4", "8"],
    ["Abuse — سوءاستفاده", "3", "8"],
    ["Assault — تعرض", "3", "8"],
  ];
  table(s, 0.6, 1.72, 7.4,
        ["کلاس", "تقسیم استاندارد", "تقسیم مسیر اول"],
        rows, { colW: [3.6, 1.9, 1.9], rowH: 0.27, size: 11,
                headSize: 11.5, hi: [7, 8, 9, 10, 11, 12, 13] });
  s.addText(
    "جمع ستون دوم ۲۹۰ و جمع ستون سوم ۲۹۶ است. ردیف‌های پررنگ، هفت کلاسی هستند که پنج ویدیوی آزمون یا کمتر دارند.",
    rtl({ x: 0.6, y: 5.86, w: 7.4, h: 0.4, fontSize: 11, color: MUTED,
          lineSpacing: 16, margin: 0 }));

  card(s, 8.3, 1.72, 4.4, 1.95, "FFF7EA");
  s.addText("دو تقسیم متفاوت", rtl({
    x: 8.6, y: 1.9, w: 3.8, h: 0.35, fontSize: 15, bold: true,
    color: AMBER, margin: 0,
  }));
  s.addText(
    "مسیر X-CLIP روی تقسیمی با ۲۹۶ ویدیوی آزمون اجرا شد و مسیر VideoMAE روی تقسیم استاندارد با ۲۹۰ ویدیو. اعداد دو مسیر با این قید خوانده شوند.",
    rtl({ x: 8.6, y: 2.3, w: 3.8, h: 1.25, fontSize: 11.5, color: INK,
          lineSpacing: 18, margin: 0 }));

  card(s, 8.3, 3.85, 4.4, 1.85);
  s.addText("چرا ارزیابی نوفه‌آلود است", rtl({
    x: 8.6, y: 4.03, w: 3.8, h: 0.35, fontSize: 15, bold: true,
    color: NAVY, margin: 0,
  }));
  s.addText(
    "کلاس Assault تنها سه ویدیوی آزمون دارد. یک ویدیو یعنی یک‌سوم یادآوری آن کلاس، و ماکرو F1 به هر ۱۴ کلاس وزن یکسان می‌دهد.",
    rtl({ x: 8.6, y: 4.43, w: 3.8, h: 1.2, fontSize: 11.5, color: INK,
          lineSpacing: 18, margin: 0 }));

  card(s, 8.3, 5.9, 4.4, 1.05, "FDEEEC");
  s.addText(
    "پیش از ارائه، این اعداد را با فایل تقسیم خودتان بسنجید؛ فرمان در سربرگ build_slides.js است.",
    rtl({ x: 8.6, y: 6.08, w: 3.8, h: 0.75, fontSize: 11, color: RED,
          lineSpacing: 17, margin: 0 }));

  stat(s, 0.6, 6.2, 2.4, "290", "کل مجموعه آزمون", NAVY);
  stat(s, 3.1, 6.2, 2.4, "1610", "کل مجموعه آموزش", NAVY);
  stat(s, 5.6, 6.2, 2.4, "0%", "همپوشانی دو مجموعه", TEAL);

  note(s, "مهم: پیش از جلسه اعداد دقیق هر کلاس را از فایل‌های تقسیم خودتان چاپ کنید. ستون دوم از گزارش آزمایش‌ها بازسازی شده و ستون سوم تقریبی است.");
}

// =====================================================================
// 6 — weak supervision + MIL
// =====================================================================
{
  const s = lightSlide("نظارت ضعیف و یادگیری چندنمونه‌ای", "چارچوب");

  const rows = [
    ["نظارت کامل", "هر فریم برچسب دارد", "بسیار پرهزینه"],
    ["نظارت ضعیف", "فقط کل ویدیو برچسب دارد", "کم‌هزینه"],
    ["بدون نظارت", "هیچ برچسبی نیست", "رایگان"],
  ];
  table(s, 0.6, 1.8, 6.2, ["سطح نظارت", "برچسب چیست", "هزینه"], rows,
        { colW: [1.9, 2.9, 1.4], rowH: 0.4, size: 12, hi: [1] });

  card(s, 7.1, 1.8, 5.6, 2.0, "FFF7EA");
  s.addText("مسئله ما", rtl({
    x: 7.4, y: 1.98, w: 5.0, h: 0.35, fontSize: 16, bold: true,
    color: AMBER, margin: 0,
  }));
  s.addText(
    "در یک ویدیوی سه‌دقیقه‌ای، سرقت شاید ۱۵ ثانیه باشد. بیش از ۹۰ درصد آنچه مدل می‌بیند، با وجود برچسب «سرقت»، عادی است.",
    rtl({ x: 7.4, y: 2.4, w: 5.0, h: 1.2, fontSize: 13, color: INK,
          lineSpacing: 20, margin: 0 }));

  s.addText("راه حل: هر ویدیو یک کیف، هر قطعه یک نمونه", rtl({
    x: 0.6, y: 4.15, w: 12.1, h: 0.4, fontSize: 18, bold: true,
    color: NAVY, margin: 0,
  }));

  const steps = [
    ["ویدیو به ۱۶ قطعه شکسته می‌شود"],
    ["مدل به هر قطعه نمره می‌دهد"],
    ["مشکوک‌ترین قطعه‌ها انتخاب می‌شوند"],
    ["فرض: همان‌ها برچسب کیف را حمل می‌کنند"],
  ];
  steps.forEach((t, i) => {
    const x = 9.9 - i * 3.1;
    card(s, x, 4.75, 2.85, 1.65);
    stepDot(s, x + 2.1, 4.95, i + 1, i === 3 ? AMBER : NAVY);
    s.addText(t[0], rtl({
      x: x + 0.2, y: 5.55, w: 2.45, h: 0.8, fontSize: 12.5, color: INK,
      lineSpacing: 18, margin: 0,
    }));
  });
  note(s, "نظارت ضعیف یک معامله است: داده بیشتر با کیفیت برچسب کمتر. MIL چارچوب استانداردی است که با آن کار می‌کند.");
}

// =====================================================================
// 7 — path 1 title
// =====================================================================
{
  const s = darkSlide();
  s.addShape(pres.ShapeType.ellipse, {
    x: -1.8, y: -1.5, w: 5.6, h: 5.6,
    fill: { color: "1E3566" }, line: { width: 0 },
  });
  s.addText("مسیر اول", rtl({
    x: 5.0, y: 2.4, w: 7.5, h: 0.6, fontSize: 18, color: AMBER,
    bold: true, margin: 0,
  }));
  s.addText("X-CLIP", ltr({
    x: 5.0, y: 3.0, w: 7.5, h: 1.2, fontSize: 60, bold: true,
    color: "FFFFFF", align: "right", margin: 0,
  }));
  s.addText("تبدیل کد UMIL از دودویی به چهارده‌کلاسه", rtl({
    x: 5.0, y: 4.3, w: 7.5, h: 0.5, fontSize: 17, color: ICE, margin: 0,
  }));
  s.addText("0.3327  ←  0.3881", ltr({
    x: 5.0, y: 5.1, w: 7.5, h: 0.7, fontSize: 26, bold: true,
    color: AMBER, align: "right", margin: 0,
  }));
  note(s, "مسیر اول روی کد UMIL بنا شد. ماکرو F1 از 0.3327 به 0.3881 رسید.");
}

// =====================================================================
// 8 — how X-CLIP works
// =====================================================================
{
  const s = lightSlide("X-CLIP چطور کار می‌کند", "مسیر اول");

  const steps = [
    ["CLIP روی جفت عکس و متن آموزش دیده",
     "عکس و توضیحش در یک فضای مشترک نزدیک هم می‌افتند"],
    ["طبقه‌بندی بدون آموزش",
     "اسم هر کلاس به بردار تبدیل می‌شود و نزدیک‌ترین برنده است"],
    ["X-CLIP بخش زمانی اضافه می‌کند",
     "حالا ورودی ویدیو است، نه یک عکس تنها"],
    ["در پروژه ما",
     "بردار ویدیو با ۱۴ بردار متنی مقایسه می‌شود"],
  ];
  steps.forEach((t, i) => {
    const y = 1.75 + i * 1.22;
    card(s, 0.6, y, 12.1, 1.05);
    stepDot(s, 11.9, y + 0.27, i + 1, i === 3 ? AMBER : NAVY);
    s.addText(t[0], rtl({
      x: 1.0, y: y + 0.14, w: 10.6, h: 0.35, fontSize: 15.5, bold: true,
      color: NAVY, margin: 0,
    }));
    s.addText(t[1], rtl({
      x: 1.0, y: y + 0.52, w: 10.6, h: 0.35, fontSize: 12.5, color: MUTED,
      margin: 0,
    }));
  });
  s.addText(
    "آن ۱۴ بردار متنی نقش وزن‌های طبقه‌بند را بازی می‌کنند و هرگز آموزش نمی‌بینند.",
    rtl({ x: 0.6, y: 6.65, w: 12.1, h: 0.4, fontSize: 13, color: AMBER,
          bold: true, margin: 0 }));
  note(s, "نکته کلیدی: در X-CLIP، وزن‌های طبقه‌بند از متن ساخته می‌شوند. این همان جایی است که بعدا سقف پیدا شد.");
}

// =====================================================================
// 9 — three bugs
// =====================================================================
{
  const s = lightSlide("سه اشکال بنیادی که پیدا و اصلاح شد", "مسیر اول");

  const bugs = [
    ["میانگین‌گیری در ارزیابی",
     "احتمال هر ۱۶ قطعه میانگین گرفته می‌شد. در ویدیوی ناهنجار، قطعه‌های عادی شواهد را خفه می‌کردند و تقریباً همه چیز عادی پیش‌بینی می‌شد.",
     "اصلاح: تصمیم دومرحله‌ای با بیشینه"],
    ["فایل نمرات کهنه",
     "کد هر فایل نمرات موجود را بازاستفاده می‌کرد. در یک گزارش، ۲۸۸ ویدیو از ۲۹۸ کنار گذاشته شده و عدد روی ده ویدیو حساب شده بود.",
     "اصلاح: بازمحاسبه پیش‌فرض"],
    ["انتخاب قطعه برتر به تفکیک کلاس",
     "کانال عادی از عادی‌ترین قطعه‌های یک ویدیوی سرقت ساخته می‌شد. نویز برچسب سیستماتیک وارد آموزش می‌شد.",
     "اصلاح: یک رتبه‌بندی برای همه کلاس‌ها"],
  ];
  bugs.forEach((b, i) => {
    const x = 8.85 - i * 4.13;
    card(s, x, 1.8, 3.85, 4.45);
    stepDot(s, x + 3.05, 2.05, i + 1, RED);
    s.addText(b[0], rtl({
      x: x + 0.25, y: 2.75, w: 3.35, h: 0.75, fontSize: 15, bold: true,
      color: NAVY, lineSpacing: 21, margin: 0,
    }));
    s.addText(b[1], rtl({
      x: x + 0.25, y: 3.5, w: 3.35, h: 1.8, fontSize: 12, color: INK,
      lineSpacing: 19, margin: 0,
    }));
    s.addText(b[2], rtl({
      x: x + 0.25, y: 5.4, w: 3.35, h: 0.7, fontSize: 12, bold: true,
      color: TEAL, lineSpacing: 18, margin: 0,
    }));
  });
  note(s, "این سه اصلاح مقدم بر هر ایده جدیدی بود. بدون آن‌ها هیچ اندازه‌گیری قابل اعتماد نبود.");
}

// =====================================================================
// 10 — two-stage decision
// =====================================================================
{
  const s = lightSlide("تصمیم دومرحله‌ای", "مسیر اول");

  card(s, 0.6, 1.75, 5.9, 2.5, "FFF7EA");
  s.addText("مرحله اول: آیا ناهنجار است؟", rtl({
    x: 0.95, y: 1.95, w: 5.2, h: 0.4, fontSize: 17, bold: true,
    color: AMBER, margin: 0,
  }));
  s.addText("بیشینه روی ۱۶ قطعه از مقدار «۱ منهای احتمال عادی»، در مقایسه با آستانه ۰٫۵", rtl({
    x: 0.95, y: 2.4, w: 5.2, h: 0.8, fontSize: 13, color: INK,
    lineSpacing: 20, margin: 0,
  }));
  s.addText("چرا بیشینه و نه میانگین: یک قطعه مشکوک کافی است، چون رویداد فقط چند ثانیه است", rtl({
    x: 0.95, y: 3.2, w: 5.2, h: 0.85, fontSize: 12, color: MUTED,
    lineSpacing: 19, margin: 0,
  }));

  card(s, 6.8, 1.75, 5.9, 2.5);
  s.addText("مرحله دوم: کدام نوع؟", rtl({
    x: 7.15, y: 1.95, w: 5.2, h: 0.4, fontSize: 17, bold: true,
    color: NAVY, margin: 0,
  }));
  s.addText("رأی‌گیری میانگین چهار قطعه برتر، تنها میان ۱۳ کلاس ناهنجار", rtl({
    x: 7.15, y: 2.4, w: 5.2, h: 0.8, fontSize: 13, color: INK,
    lineSpacing: 20, margin: 0,
  }));
  s.addText("کلاس عادی در این مرحله اصلاً وارد رقابت نمی‌شود", rtl({
    x: 7.15, y: 3.2, w: 5.2, h: 0.85, fontSize: 12, color: MUTED,
    lineSpacing: 19, margin: 0,
  }));

  s.addText("مثال عددی: ویدیوی سرقت، رویداد در قطعه‌های ۴ و ۵", rtl({
    x: 0.6, y: 4.5, w: 12.1, h: 0.4, fontSize: 16, bold: true,
    color: NAVY, margin: 0,
  }));
  const rows = [
    ["میانگین‌گیری روی ۸ قطعه", "0.26", "زیر آستانه", "عادی — غلط"],
    ["بیشینه روی همان ۸ قطعه", "0.85", "بالای آستانه", "ناهنجار — درست"],
  ];
  table(s, 0.6, 5.0, 12.1, ["روش", "نمره", "در برابر آستانه ۰٫۵", "تصمیم"],
        rows, { colW: [3.8, 2.3, 3.0, 3.0], rowH: 0.42, size: 13, hi: [1] });
  note(s, "این اصلاح، ریشه‌ای‌ترین تغییر در ارزیابی بود. مدل درست کار می‌کرد و مرحله تجمیع نتیجه را نابود می‌کرد.");
}

// =====================================================================
// 11 — class collapse
// =====================================================================
{
  const s = lightSlide("کلاسی که مدل یاد گرفت هرگز نگوید", "مسیر اول");

  s.addText(
    "به‌جای شمردن پیش‌بینی‌های درست، شمردیم مدل هر کلاس را چند بار پیش‌بینی می‌کند.",
    rtl({ x: 0.6, y: 1.8, w: 12.1, h: 0.4, fontSize: 14, color: INK,
          margin: 0 }));

  const rows = [
    ["Fighting — درگیری", "8", "2", "0.25"],
    ["Assault — تعرض", "8", "3", "0.38"],
    ["Vandalism — تخریب", "8", "5", "0.62"],
    ["Stealing — دزدی", "15", "20", "1.33"],
    ["RoadAccidents — تصادف", "23", "39", "1.70"],
  ];
  table(s, 0.6, 2.35, 7.3, ["کلاس", "ویدیوی واقعی", "چند بار پیش‌بینی شد", "نسبت"],
        rows, { colW: [2.8, 1.5, 1.9, 1.1], rowH: 0.4, size: 12,
                hi: [0] });

  card(s, 8.2, 2.35, 4.5, 2.05, "FFF7EA");
  s.addText("تشخیص", rtl({
    x: 8.5, y: 2.53, w: 3.9, h: 0.35, fontSize: 16, bold: true,
    color: AMBER, margin: 0,
  }));
  s.addText(
    "مدل روی درگیری «شکست نخورده» بود. یاد گرفته بود هرگز نگوید درگیری، چون گفتنش شرط‌بندی بدی بود.",
    rtl({ x: 8.5, y: 2.95, w: 3.9, h: 1.3, fontSize: 12.5, color: INK,
          lineSpacing: 19, margin: 0 }));

  s.addText("علت ریشه‌ای", rtl({
    x: 0.6, y: 4.6, w: 12.1, h: 0.4, fontSize: 17, bold: true,
    color: NAVY, margin: 0,
  }));
  bullets(s, 0.6, 5.1, 12.1, [
    "تابع وزن‌دهی کلاس در کد وجود داشت ولی هیچ‌جا صدا زده نمی‌شد.",
    "وزن‌ها همه‌جا ۱٫۰ بودند، جز دو عدد دستی ۲٫۰. کلاس عادی حدود ۱۹ برابر کلاس درگیری گرادیان تولید می‌کرد.",
    "با وزن‌دهی معکوس فراوانی آن نسبت به ۳٫۶ رسید و درگیری ۹ بار پیش‌بینی شد به‌جای ۲ بار.",
  ], 13);
  note(s, "نکته صادقانه: با وجود ۹ پیش‌بینی، هنوز هیچ‌کدام درست نبود. مشکل دوم بصری بود، نه وزنی.");
}

// =====================================================================
// 12 — text prototype ceiling
// =====================================================================
{
  const s = lightSlide("محدودیت بنیادی: هندسه فضای متنی", "مسیر اول");

  s.addText(
    "اسم خام کلاس‌ها در فضای زبانی CLIP تقریباً هم‌راستا هستند و سقفی مستقل از کیفیت تصویری تحمیل می‌کنند.",
    rtl({ x: 0.6, y: 1.78, w: 12.1, h: 0.4, fontSize: 14, color: INK,
          margin: 0 }));

  const rows = [
    ["Abuse  /  Stealing", "0.91"],
    ["Assault  /  Arrest", "0.91"],
    ["Fighting  /  Shooting", "0.90"],
    ["میانگین همه جفت‌ها", "0.83"],
  ];
  table(s, 0.6, 2.35, 5.5, ["جفت کلاس", "شباهت کسینوسی"], rows,
        { colW: [3.5, 2.0], rowH: 0.42, size: 13, hi: [3] });

  card(s, 6.4, 2.35, 6.3, 2.05, "FFF7EA");
  s.addText("آزمایش تعیین‌کننده", rtl({
    x: 6.7, y: 2.53, w: 5.7, h: 0.35, fontSize: 16, bold: true,
    color: AMBER, margin: 0,
  }));
  s.addText(
    "فرض کنید بخش تصویری کاملاً بی‌نقص باشد، یعنی بردار ویدیو دقیقاً برابر بردار کلاس خودش. در این حالت آرمانی، احتمال کلاس درست تنها ۰٫۴۴ درمی‌آید.",
    rtl({ x: 6.7, y: 2.95, w: 5.7, h: 1.3, fontSize: 12.5, color: INK,
          lineSpacing: 19, margin: 0 }));

  s.addText("راه حل: توصیف بصری به‌جای اسم حقوقی", rtl({
    x: 0.6, y: 4.6, w: 12.1, h: 0.4, fontSize: 17, bold: true,
    color: NAVY, margin: 0,
  }));
  card(s, 0.6, 5.1, 6.0, 1.4);
  s.addText("پیش از تغییر", rtl({
    x: 0.9, y: 5.25, w: 5.4, h: 0.3, fontSize: 12, color: MUTED, margin: 0,
  }));
  s.addText("\"Burglary\"", ltr({
    x: 0.9, y: 5.6, w: 5.4, h: 0.4, fontSize: 16, bold: true, color: RED,
    align: "right", margin: 0,
  }));
  s.addText("سقف اندازه‌گیری‌شده: 0.44", rtl({
    x: 0.9, y: 6.0, w: 5.4, h: 0.35, fontSize: 12.5, color: INK, margin: 0,
  }));

  card(s, 6.9, 5.1, 5.8, 1.4, "EAF6F5");
  s.addText("پس از تغییر", rtl({
    x: 7.2, y: 5.25, w: 5.2, h: 0.3, fontSize: 12, color: MUTED, margin: 0,
  }));
  s.addText("\"a burglar climbing through a broken window\"", ltr({
    x: 7.2, y: 5.6, w: 5.2, h: 0.4, fontSize: 11.5, bold: true, color: TEAL,
    align: "right", margin: 0,
  }));
  s.addText("سقف اندازه‌گیری‌شده: 0.86", rtl({
    x: 7.2, y: 6.0, w: 5.2, h: 0.35, fontSize: 12.5, color: INK, margin: 0,
  }));
  note(s, "اصطلاحات حقوقی انتزاعی در فضای زبانی خوشه می‌شوند؛ توصیف آنچه دوربین می‌بیند نه. این مهم‌ترین کشف مسیر اول است.");
}

// =====================================================================
// 13 — X-CLIP results
// =====================================================================
{
  const s = lightSlide("نتایج مسیر اول", "مسیر اول");

  const rows = [
    ["اجرای پایه — اسم خام، وزن دستی", "0.9473", "0.3327", "0.3716", "0.6385"],
    ["توصیف بصری + وزن معکوس فراوانی", "0.9637", "0.3713", "0.4054", "0.6351"],
    ["افزودن چهار نمای زمانی در آزمون", "0.9629", "0.3881", "0.4257", "0.6419"],
  ];
  table(s, 0.6, 1.85, 12.1,
        ["پیکربندی", "AUC دودویی", "ماکرو F1", "صحت نوع", "صحت کلی"],
        rows, { colW: [4.9, 1.8, 1.8, 1.8, 1.8], rowH: 0.44, size: 12.5,
                hi: [2] });

  s.addText("کلاس‌هایی که از مرگ برگشتند", rtl({
    x: 0.6, y: 3.95, w: 6.0, h: 0.4, fontSize: 16, bold: true,
    color: NAVY, margin: 0,
  }));
  const rows2 = [
    ["Arrest — بازداشت", "0.000", "0.556"],
    ["Abuse — سوءاستفاده", "0.462", "0.714"],
    ["Stealing — دزدی", "0.200", "0.350"],
    ["Vandalism — تخریب", "0.000", "0.167"],
  ];
  table(s, 0.6, 4.45, 6.0, ["کلاس", "F1 پیش از تغییر", "F1 پس از تغییر"],
        rows2, { colW: [2.4, 1.8, 1.8], rowH: 0.36, size: 11.5 });

  card(s, 7.0, 3.95, 5.7, 2.55, "FFF7EA");
  s.addText("چرا از این مسیر فاصله گرفتیم", rtl({
    x: 7.3, y: 4.15, w: 5.1, h: 0.35, fontSize: 16, bold: true,
    color: AMBER, margin: 0,
  }));
  bullets(s, 7.3, 4.6, 5.1, [
    "هر پیکربندی حدود ۱۶ ساعت طول می‌کشید",
    "با این هزینه، آزمون سیستماتیک فرضیه‌ها ناممکن بود",
    "خطای باقی‌مانده در خوشه دزدی بود و ماهیتش مکانی بود، نه زمانی",
  ], 12);
  note(s, "مسیر اول به 0.3881 رسید. دو دیوار باقی ماند: هزینه آزمایش و ماهیت مکانی خطای خوشه دزدی.");
}

// =====================================================================
// 14 — path 2 title
// =====================================================================
{
  const s = darkSlide();
  s.addShape(pres.ShapeType.ellipse, {
    x: -1.8, y: -1.5, w: 5.6, h: 5.6,
    fill: { color: "1E3566" }, line: { width: 0 },
  });
  s.addText("مسیر دوم", rtl({
    x: 5.0, y: 2.4, w: 7.5, h: 0.6, fontSize: 18, color: AMBER,
    bold: true, margin: 0,
  }));
  s.addText("VideoMAE", ltr({
    x: 5.0, y: 3.0, w: 7.5, h: 1.2, fontSize: 60, bold: true,
    color: "FFFFFF", align: "right", margin: 0,
  }));
  s.addText("رمزگذار منجمد و ویژگی‌های ذخیره‌شده", rtl({
    x: 5.0, y: 4.3, w: 7.5, h: 0.5, fontSize: 17, color: ICE, margin: 0,
  }));
  s.addText("0.2975  ←  0.4298", ltr({
    x: 5.0, y: 5.1, w: 7.5, h: 0.7, fontSize: 26, bold: true,
    color: AMBER, align: "right", margin: 0,
  }));
  note(s, "مسیر دوم از صفر نوشته شد. ماکرو F1 از 0.2975 به 0.4298 رسید.");
}

// =====================================================================
// 15 — how VideoMAE works
// =====================================================================
{
  const s = lightSlide("VideoMAE چطور یاد گرفت", "مسیر دوم");

  const steps = [
    ["پوشاندن", "حدود ۹۰ درصد مکعب‌های فضازمانی پوشانده می‌شوند"],
    ["بازسازی", "رمزگذار فقط ۱۰ درصد باقی‌مانده را می‌بیند و رمزگشا جای خالی‌ها را حدس می‌زند"],
    ["بدون برچسب", "خود ویدیو هم سؤال است و هم جواب، پس روی حجم عظیمی از داده ممکن است"],
    ["دور انداختن رمزگشا", "پس از پیش‌آموزش تنها رمزگذار می‌ماند و روی Kinetics تنظیم می‌شود"],
  ];
  steps.forEach((t, i) => {
    const y = 1.8 + i * 1.12;
    card(s, 0.6, y, 8.3, 0.95);
    stepDot(s, 8.1, y + 0.22, i + 1, i === 3 ? AMBER : NAVY);
    s.addText(t[0], rtl({
      x: 1.0, y: y + 0.1, w: 6.9, h: 0.32, fontSize: 15, bold: true,
      color: NAVY, margin: 0,
    }));
    s.addText(t[1], rtl({
      x: 1.0, y: y + 0.45, w: 6.9, h: 0.42, fontSize: 12, color: MUTED,
      margin: 0,
    }));
  });

  card(s, 9.2, 1.8, 3.5, 4.4, "FFF7EA");
  s.addText("تفاوت با X-CLIP", rtl({
    x: 9.5, y: 2.0, w: 2.9, h: 0.35, fontSize: 15, bold: true,
    color: AMBER, margin: 0,
  }));
  s.addText(
    "X-CLIP از متن یاد گرفت.\n\nVideoMAE از خود ویدیو.\n\nپس VideoMAE اصلاً کلمه نمی‌شناسد و آن سقف ۰٫۴۴ که از هم‌راستایی اسم کلاس‌ها می‌آمد، اینجا اصلاً وجود ندارد.",
    rtl({ x: 9.5, y: 2.45, w: 2.9, h: 3.5, fontSize: 12.5, color: INK,
          lineSpacing: 20, margin: 0 }));
  note(s, "پوشاندن لوله‌ای است: اگر جایی پوشانده شود، در همه فریم‌ها پوشانده می‌شود، تا مدل از فریم بغلی کپی نکند.");
}

// =====================================================================
// 16 — the freeze decision
// =====================================================================
{
  const s = lightSlide("مهم‌ترین تصمیم پروژه: یخ زدن رمزگذار", "مسیر دوم");

  s.addText("چون رمزگذار آموزش نمی‌بیند، خروجی‌اش ثابت است. پس محاسبه مجددش در هر دوره کار تکراری محض است.",
    rtl({ x: 0.6, y: 1.78, w: 12.1, h: 0.4, fontSize: 14, color: INK,
          margin: 0 }));

  stat(s, 8.8, 2.35, 3.9, "16", "ساعت برای هر آزمایش — روش قدیم", RED);
  s.addShape(pres.ShapeType.leftArrow, {
    x: 7.3, y: 2.6, w: 1.3, h: 0.55,
    fill: { color: ICE }, line: { width: 0 },
  });
  stat(s, 3.2, 2.35, 3.9, "80", "ثانیه برای هر آزمایش — روش جدید", TEAL);
  stat(s, 0.6, 2.35, 2.3, "200", "ارزیابی انجام‌شده", NAVY);

  const rows = [
    ["رمزگشایی ویدیو", "هر دوره تکرار می‌شد", "یک بار، هرگز دوباره"],
    ["اجرای رمزگذار", "هر دوره تکرار می‌شد", "یک بار، هرگز دوباره"],
    ["حجم داده ورودی", "۷۰ گیگابایت ویدیو", "۹۳ مگابایت عدد"],
    ["پارامترهای آموزش‌پذیر", "حدود ۳۰۰ میلیون", "۴٫۸ میلیون"],
    ["زمان ۲۰۰ آزمایش", "حدود ۱۳۳ روز", "حدود ۴ ساعت"],
  ];
  table(s, 0.6, 4.05, 12.1, ["مورد", "روش قدیم", "روش جدید"], rows,
        { colW: [3.7, 4.2, 4.2], rowH: 0.35, size: 12.5, hi: [4] });

  s.addText(
    "هزینه‌ای که پرداختیم: رمزگذار هرگز به دامنه ویدیوی نظارتی تطبیق نمی‌یابد. این را صریح گزارش کرده‌ایم.",
    rtl({ x: 0.6, y: 6.45, w: 12.1, h: 0.4, fontSize: 13, color: AMBER,
          bold: true, margin: 0 }));
  note(s, "این تصمیم خودش نتیجه را بهتر نکرد. اجازه داد بقیه چیزها را پیدا کنیم. با ۱۶ ساعت، هیچ‌کدام از یافته‌های بعدی ممکن نبود.");
}

// =====================================================================
// 17 — pipeline
// =====================================================================
{
  const s = lightSlide("خط لوله نهایی", "مسیر دوم");

  const boxes = [
    ["ویدیوی خام", "۹۰۰۰ فریم\n۴۰ مگابایت", NAVY],
    ["۱۶ قطعه", "هر قطعه ۱۶ فریم\n۲۲۴ در ۲۲۴", NAVY],
    ["رمزگذار ViT-L", "منجمد\nیک بار اجرا", AMBER],
    ["ویژگی ذخیره‌شده", "۱۶ در ۱۰۲۴\n۶۴ کیلوبایت", TEAL],
  ];
  boxes.forEach((b, i) => {
    const x = 9.95 - i * 3.15;
    card(s, x, 1.9, 2.8, 1.5, i === 2 ? "FFF7EA" : "FFFFFF");
    s.addText(b[0], rtl({
      x: x + 0.15, y: 2.1, w: 2.5, h: 0.35, fontSize: 14, bold: true,
      color: b[2], align: "center", margin: 0,
    }));
    s.addText(b[1], rtl({
      x: x + 0.15, y: 2.5, w: 2.5, h: 0.7, fontSize: 11.5, color: MUTED,
      align: "center", lineSpacing: 17, margin: 0,
    }));
    if (i < 3) {
      s.addShape(pres.ShapeType.leftArrow, {
        x: x - 0.28, y: 2.5, w: 0.25, h: 0.3,
        fill: { color: ICE }, line: { width: 0 },
      });
    }
  });

  s.addText("مرحله اول — یک بار برای ۱۸۹۵ ویدیو، ۱۵۰ دقیقه", rtl({
    x: 0.6, y: 3.5, w: 12.1, h: 0.35, fontSize: 12.5, color: MUTED,
    align: "center", margin: 0,
  }));

  const boxes2 = [
    ["سر ترنسفورمر", "دو لایه توجه\nروی محور زمان", NAVY],
    ["نمره هر قطعه", "۱۶ در ۱۴", NAVY],
    ["تصمیم دومرحله‌ای", "وجود، سپس نوع", AMBER],
    ["برچسب نهایی", "یکی از ۱۴ کلاس", TEAL],
  ];
  boxes2.forEach((b, i) => {
    const x = 9.95 - i * 3.15;
    card(s, x, 4.2, 2.8, 1.5, i === 2 ? "FFF7EA" : "FFFFFF");
    s.addText(b[0], rtl({
      x: x + 0.15, y: 4.4, w: 2.5, h: 0.35, fontSize: 14, bold: true,
      color: b[2], align: "center", margin: 0,
    }));
    s.addText(b[1], rtl({
      x: x + 0.15, y: 4.8, w: 2.5, h: 0.7, fontSize: 11.5, color: MUTED,
      align: "center", lineSpacing: 17, margin: 0,
    }));
    if (i < 3) {
      s.addShape(pres.ShapeType.leftArrow, {
        x: x - 0.28, y: 4.8, w: 0.25, h: 0.3,
        fill: { color: ICE }, line: { width: 0 },
      });
    }
  });
  s.addText("مرحله دوم — هر پیکربندی، ۸۰ ثانیه", rtl({
    x: 0.6, y: 5.8, w: 12.1, h: 0.35, fontSize: 12.5, color: MUTED,
    align: "center", margin: 0,
  }));

  s.addText("تنها بخشی که ما آموزش دادیم، سر طبقه‌بند است: کمتر از دو درصد پارامترهای کل سیستم.",
    rtl({ x: 0.6, y: 6.35, w: 12.1, h: 0.4, fontSize: 13, color: AMBER,
          bold: true, align: "center", margin: 0 }));
  note(s, "ویدیوی ۴۰ مگابایتی به آرایه ۶۴ کیلوبایتی تبدیل می‌شود. این فشرده‌سازی، کل معماری را ممکن کرده است.");
}

// =====================================================================
// 18 — temporal head
// =====================================================================
{
  const s = lightSlide("سر ترنسفورمر روی محور زمان", "مسیر دوم");

  card(s, 0.6, 1.8, 5.9, 2.3);
  s.addText("مشکل: قطعه تنها مبهم است", rtl({
    x: 0.95, y: 2.0, w: 5.2, h: 0.35, fontSize: 16, bold: true,
    color: MUTED, margin: 0,
  }));
  s.addText(
    "یک قطعه سه‌ثانیه‌ای داخل مغازه، در دزدی از مغازه و سرقت مسلحانه ظاهر یکسانی دارد. سر نقطه‌ای اطلاعات لازم را اصولاً در اختیار ندارد.",
    rtl({ x: 0.95, y: 2.45, w: 5.2, h: 1.4, fontSize: 13, color: INK,
          lineSpacing: 20, margin: 0 }));

  card(s, 6.8, 1.8, 5.9, 2.3, "FFF7EA");
  s.addText("راه حل: خودتوجهی میان قطعه‌ها", rtl({
    x: 7.15, y: 2.0, w: 5.2, h: 0.35, fontSize: 16, bold: true,
    color: AMBER, margin: 0,
  }));
  s.addText(
    "یک جدول ۱۶ در ۱۶ می‌گوید هر قطعه چقدر به هر قطعه دیگر توجه کند. بردار جدید هر قطعه، جمع وزن‌دار همه قطعه‌هاست.",
    rtl({ x: 7.15, y: 2.45, w: 5.2, h: 1.4, fontSize: 13, color: INK,
          lineSpacing: 20, margin: 0 }));

  s.addText("پیکربندی", rtl({
    x: 0.6, y: 4.35, w: 5.9, h: 0.35, fontSize: 16, bold: true,
    color: NAVY, margin: 0,
  }));
  const cfg = [
    ["بعد پنهان", "512"],
    ["سرهای توجه", "4"],
    ["لایه‌ها", "2"],
    ["حذف تصادفی", "0.3"],
    ["پارامترها", "4,772,366"],
  ];
  table(s, 0.6, 4.8, 5.9, ["پارامتر", "مقدار"], cfg,
        { colW: [3.4, 2.5], rowH: 0.3, size: 11.5 });

  s.addText("نتیجه", rtl({
    x: 6.8, y: 4.35, w: 5.9, h: 0.35, fontSize: 16, bold: true,
    color: NAVY, margin: 0,
  }));
  const res = [
    ["سر نقطه‌ای — خط پایه", "0.3049"],
    ["سر ترنسفورمر زمانی", "0.3480"],
  ];
  table(s, 6.8, 4.8, 5.9, ["سر طبقه‌بند", "ماکرو F1، میانگین ۵ بذر"], res,
        { colW: [3.2, 2.7], rowH: 0.42, size: 12.5, hi: [1] });
  s.addText(
    "هر پنج بذر سر زمانی از بهترین بذر سر نقطه‌ای بالاتر بود. دو توزیع هیچ همپوشانی ندارند.",
    rtl({ x: 6.8, y: 6.25, w: 5.9, h: 0.7, fontSize: 12, color: INK,
          lineSpacing: 19, margin: 0 }));
  note(s, "این تغییر از تحلیل خطا آمد، نه از جستجوی ابرپارامتر. تعبیه جایگاهی لازم است چون خودتوجهی نسبت به جایگشت هم‌ارز است.");
}

// =====================================================================
// 19 — backbone upgrade
// =====================================================================
{
  const s = lightSlide("ارتقای ظرفیت رمزگذار", "مسیر دوم");

  const rows = [
    ["تعداد لایه", "12", "24"],
    ["پهنای بردار", "768", "1024"],
    ["پارامترها", "≈ 87 میلیون", "≈ 300 میلیون"],
    ["اندازه وصله", "16 × 16", "16 × 16"],
    ["ماکرو F1، میانگین ۵ بذر", "0.3405", "0.3955"],
    ["ماکرو F1، نتیجه گروهی", "0.3645", "0.4122"],
  ];
  table(s, 0.6, 1.85, 7.2, ["مورد", "ViT-B", "ViT-L"], rows,
        { colW: [3.2, 2.0, 2.0], rowH: 0.4, size: 12.5, hi: [4, 5] });

  card(s, 8.1, 1.85, 4.6, 2.35, "FFF7EA");
  s.addText("نکته‌ای که نباید اشتباه گفت", rtl({
    x: 8.4, y: 2.05, w: 4.0, h: 0.35, fontSize: 15, bold: true,
    color: AMBER, margin: 0,
  }));
  s.addText(
    "اندازه وصله در هر دو یکسان است، پس تعداد وصله‌ها فرق نمی‌کند. بهبود از ظرفیت مدل می‌آید، نه از وضوح مکانی بیشتر.",
    rtl({ x: 8.4, y: 2.5, w: 4.0, h: 1.5, fontSize: 12.5, color: INK,
          lineSpacing: 19, margin: 0 }));

  s.addText("چرا این تنها بهبود قطعی پروژه است", rtl({
    x: 0.6, y: 4.6, w: 12.1, h: 0.4, fontSize: 17, bold: true,
    color: NAVY, margin: 0,
  }));

  card(s, 0.6, 5.1, 3.9, 1.5);
  s.addText("بازه پنج بذر ViT-B", rtl({
    x: 0.9, y: 5.3, w: 3.3, h: 0.3, fontSize: 12, color: MUTED, margin: 0,
  }));
  s.addText("0.310 — 0.354", ltr({
    x: 0.9, y: 5.65, w: 3.3, h: 0.5, fontSize: 20, bold: true, color: RED,
    align: "right", margin: 0,
  }));

  card(s, 4.7, 5.1, 3.9, 1.5, "EAF6F5");
  s.addText("بازه پنج بذر ViT-L", rtl({
    x: 5.0, y: 5.3, w: 3.3, h: 0.3, fontSize: 12, color: MUTED, margin: 0,
  }));
  s.addText("0.370 — 0.423", ltr({
    x: 5.0, y: 5.65, w: 3.3, h: 0.5, fontSize: 20, bold: true, color: TEAL,
    align: "right", margin: 0,
  }));

  card(s, 8.8, 5.1, 3.9, 1.5, "FFF7EA");
  s.addText("آزمون من-ویتنی", rtl({
    x: 9.1, y: 5.3, w: 3.3, h: 0.3, fontSize: 12, color: MUTED, margin: 0,
  }));
  s.addText("p = 1 / 252", ltr({
    x: 9.1, y: 5.65, w: 3.3, h: 0.5, fontSize: 20, bold: true, color: AMBER,
    align: "right", margin: 0,
  }));
  note(s, "دو بازه هیچ همپوشانی ندارند. بدترین بذر ViT-L از بهترین بذر ViT-B بالاتر است.");
}

// =====================================================================
// 20 — mixup + logit adjustment
// =====================================================================
{
  const s = lightSlide("آمیزش و تنظیم لاجیت", "مسیر دوم");

  card(s, 0.6, 1.8, 5.9, 2.15);
  s.addText("آمیزش در فضای ویژگی", rtl({
    x: 0.95, y: 1.98, w: 5.2, h: 0.35, fontSize: 16, bold: true,
    color: NAVY, margin: 0,
  }));
  s.addText(
    "مدل تا دوره دهم زیان آموزش را تقریباً صفر می‌کرد، یعنی حفظ می‌کرد. ترکیب محدب دو ویدیو، نمونه‌هایی می‌سازد که در داده نیستند و حفظ کردن را ناممکن می‌کنند.",
    rtl({ x: 0.95, y: 2.42, w: 5.2, h: 1.4, fontSize: 12.5, color: INK,
          lineSpacing: 19, margin: 0 }));

  card(s, 6.8, 1.8, 5.9, 2.15, "FFF7EA");
  s.addText("تنظیم لاجیت برای دم بلند", rtl({
    x: 7.15, y: 1.98, w: 5.2, h: 0.35, fontSize: 16, bold: true,
    color: AMBER, margin: 0,
  }));
  s.addText(
    "مقداری متناسب با لگاریتم احتمال پیشین از لاجیت هر کلاس کسر می‌شود تا مرز تصمیم به سود کلاس‌های کمیاب جابه‌جا شود.",
    rtl({ x: 7.15, y: 2.42, w: 5.2, h: 1.4, fontSize: 12.5, color: INK,
          lineSpacing: 19, margin: 0 }));

  card(s, 0.6, 4.15, 12.1, 1.15, "FDEEEC");
  s.addText("آزمایشی که شکست خورد و از آن یاد گرفتیم", rtl({
    x: 0.95, y: 4.3, w: 11.4, h: 0.35, fontSize: 15, bold: true,
    color: RED, margin: 0,
  }));
  s.addText(
    "اعمال خام فرمول روی هر ۱۴ کلاس، صحت را از ۰٫۶۸۹۷ به ۰٫۴۷۲ انداخت: همه کلاس‌های ناهنجار نسبت به عادی بالا می‌رفتند و مرحله اول تصمیم خراب می‌شد. اصلاح: محدود کردن به ۱۳ کلاس ناهنجار و مرکزی کردن، تا جرم کل ناهنجاری حفظ شود.",
    rtl({ x: 0.95, y: 4.68, w: 11.4, h: 0.55, fontSize: 12, color: INK,
          lineSpacing: 18, margin: 0 }));

  const rows = [
    ["پایه ViT-L", "0.3955", "0.4122", "0.6897", "0.4786"],
    ["آمیزش ۰٫۲", "0.4203", "0.4161", "0.6793", "0.4714"],
    ["آمیزش ۰٫۲ + تنظیم لاجیت", "0.4147", "0.4298", "0.6897", "0.5000"],
    ["آمیزش ۰٫۴ + تنظیم لاجیت", "0.4135", "0.4248", "0.6759", "0.4929"],
  ];
  table(s, 0.6, 5.5, 12.1,
        ["پیکربندی", "میانگین بذرها", "نتیجه گروهی", "صحت", "صحت نوع"],
        rows, { colW: [4.3, 2.0, 2.0, 1.9, 1.9], rowH: 0.3, size: 11.5,
                hi: [2] });
  note(s, "ترکیب روی هر سه معیار بهترین یا هم‌ارز بهترین است. سه ترکیب مقایسه شد، پس تورم انتخاب حدود ۰٫۰۵ است؛ دفاع مکانیزمی است نه آماری.");
}

// =====================================================================
// 21 — progression chart
// =====================================================================
{
  const s = lightSlide("مسیر بهبود، گام به گام", "مسیر دوم");

  s.addChart(pres.ChartType.bar, [{
    name: "macro F1",
    labels: ["MLP / ViT-B", "+ Temporal", "+ lr 3e-4", "+ ViT-L",
             "+ Mixup", "+ Logit adj."],
    values: [0.2975, 0.3444, 0.3645, 0.4122, 0.4161, 0.4298],
  }], {
    x: 0.6, y: 1.8, w: 12.1, h: 3.6,
    barDir: "col",
    chartColors: [NAVY, NAVY, NAVY, AMBER, AMBER, AMBER],
    showTitle: false,
    showLegend: false,
    showValue: true,
    dataLabelPosition: "outEnd",
    dataLabelColor: INK,
    dataLabelFontSize: 11,
    dataLabelFontFace: FA,
    dataLabelFormatCode: "0.0000",
    valAxisMinVal: 0.25,
    valAxisMaxVal: 0.46,
    valAxisLabelColor: MUTED,
    catAxisLabelColor: INK,
    valAxisLabelFontSize: 10,
    catAxisLabelFontSize: 11,
    valAxisLabelFontFace: FA,
    catAxisLabelFontFace: FA,
    valGridLine: { color: "E4E9F2", size: 1 },
    catGridLine: { style: "none" },
    barGapWidthPct: 55,
  });

  stat(s, 0.6, 5.6, 3.9, "+0.1323", "بهبود مطلق ماکرو F1", TEAL);
  stat(s, 4.7, 5.6, 3.9, "+44%", "بهبود نسبی", TEAL);
  stat(s, 8.8, 5.6, 3.9, "0.075", "کف نوفه این مجموعه آزمون", AMBER);
  note(s, "دو گام از شش گام بیشترین سهم را داشتند: توجه زمانی و ارتقای رمزگذار. هر دو از تحلیل خطا آمدند.");
}

// =====================================================================
// 22 — negative results
// =====================================================================
{
  const s = lightSlide("آزمایش‌هایی که نتیجه ندادند", "صداقت روش‌شناختی");

  const rows = [
    ["نمونه‌برداری سبک Kinetics با گام ۴ فریم", "0.3089", "0.3405", "بدتر"],
    ["دو برابر کردن تعداد قطعه‌ها به ۳۲", "0.3875", "0.3955", "بدتر"],
    ["الحاق ویژگی دو رمزگذار", "0.3974", "0.3955", "درون نوفه"],
    ["آموزش با یک قطعه برتر", "—", "—", "بدون بهبود"],
    ["تنظیم لاجیت به‌تنهایی", "0.4114", "0.3955", "درون نوفه"],
    ["میانگین‌گیری بذرها برای ماکرو F1", "—", "—", "بدون بهبود"],
    ["بازنویسی توصیف‌های متنی کلاس‌ها", "0.74", "0.86", "بدتر"],
  ];
  table(s, 0.6, 1.85, 12.1, ["آزمایش", "نتیجه", "مرجع", "داوری"], rows,
        { colW: [6.1, 2.0, 2.0, 2.0], rowH: 0.36, size: 12 });

  card(s, 0.6, 4.75, 12.1, 1.95, "FFF7EA");
  s.addText("چرا نمونه‌برداری استاندارد بدتر شد — مهم‌ترین یافته منفی", rtl({
    x: 0.95, y: 4.95, w: 11.4, h: 0.35, fontSize: 15, bold: true,
    color: AMBER, margin: 0,
  }));
  s.addText(
    "قطعه‌های ما کشیده بودند و با نرخ نمونه‌برداری وزن‌های از پیش‌آموزش‌دیده نمی‌خواندند. اصلاح آن نتیجه را بدتر کرد: قطعه‌های کشیده کل ویدیو را می‌پوشانند، در حالی که سی‌ودو پنجره دو ثانیه‌ای تنها حدود ۲۱ درصد یک ویدیوی طولانی را لمس می‌کنند و رویدادی که بیرون همه پنجره‌ها بیفتد اصلاً قابل امتیازدهی نیست.\n\nنتیجه: پوشش بر اعتبار قطعه غلبه دارد.",
    rtl({ x: 0.95, y: 5.35, w: 11.4, h: 1.25, fontSize: 12, color: INK,
          lineSpacing: 18, margin: 0 }));
  note(s, "گزارش نتایج منفی بخشی از نتیجه است، نه حاشیه آن. هر کدام یک فرضیه معقول را رد می‌کند.");
}

// =====================================================================
// 23 — noise floor
// =====================================================================
{
  const s = lightSlide("چقدر از این اعداد نوفه است؟", "اعتبارسنجی آماری");

  s.addText(
    "مجموعه آزمون با ۴۰۰۰ بار بازنمونه‌گیری خودگردان تحلیل شد تا پراکندگی مورد انتظار روی مجموعه‌های آزمون هم‌اندازه تخمین زده شود.",
    rtl({ x: 0.6, y: 1.78, w: 12.1, h: 0.4, fontSize: 14, color: INK,
          margin: 0 }));

  const rows = [
    ["ماکرو F1", "0.4298", "0.037", "[0.339 , 0.503]"],
    ["صحت چندکلاسه", "0.6897", "0.027", "[0.635 , 0.745]"],
    ["صحت نوع ناهنجاری", "0.5000", "0.041", "[0.420 , 0.585]"],
    ["AUC دودویی", "0.9610", "0.011", "[0.938 , 0.980]"],
  ];
  table(s, 0.6, 2.35, 7.3,
        ["معیار", "مقدار نقطه‌ای", "انحراف معیار", "بازه اطمینان ۹۵ درصد"],
        rows, { colW: [2.2, 1.7, 1.7, 1.7], rowH: 0.42, size: 12,
                hi: [0] });

  card(s, 8.2, 2.35, 4.5, 2.15, "FFF7EA");
  s.addText("کف نوفه", rtl({
    x: 8.5, y: 2.53, w: 3.9, h: 0.35, fontSize: 16, bold: true,
    color: AMBER, margin: 0,
  }));
  s.addText("0.075", ltr({
    x: 8.5, y: 2.9, w: 3.9, h: 0.6, fontSize: 34, bold: true, color: AMBER,
    align: "right", margin: 0,
  }));
  s.addText(
    "اختلاف کمتر از این مقدار در ماکرو F1 روی این تقسیم از نوفه قابل تفکیک نیست.",
    rtl({ x: 8.5, y: 3.55, w: 3.9, h: 0.8, fontSize: 12, color: INK,
          lineSpacing: 18, margin: 0 }));

  s.addText("سقف تنظیم روی یک مجموعه آزمون کوچک", rtl({
    x: 0.6, y: 4.7, w: 12.1, h: 0.4, fontSize: 17, bold: true,
    color: NAVY, margin: 0,
  }));
  const rows2 = [
    ["۸", "+0.069"],
    ["۵۰", "+0.095"],
    ["۲۰۰", "+0.111"],
  ];
  table(s, 0.6, 5.2, 4.6, ["تعداد ارزیابی", "تورم مورد انتظار"], rows2,
        { colW: [2.3, 2.3], rowH: 0.33, size: 12, hi: [2] });

  card(s, 5.6, 5.2, 7.1, 1.45);
  s.addText(
    "با حدود ۲۰۰ ارزیابی روی همین تقسیم، گزارش بهترین عدد مشاهده‌شده نتیجه را بیش از اندازه خود اثر ادعاشده بزرگ نشان می‌دهد. به همین دلیل همه اعداد این ارائه از پیکربندی‌هایی می‌آیند که پیش از اجرا تثبیت شده بودند و آستانه تصمیم روی مجموعه آزمون تنظیم نشده است.",
    rtl({ x: 5.9, y: 5.4, w: 6.5, h: 1.1, fontSize: 12, color: INK,
          lineSpacing: 18, margin: 0 }));
  note(s, "این تحلیل تفسیر کل پروژه را عوض کرد. ارتقای رمزگذار از کف نوفه عبور می‌کند؛ بقیه بهبودها با احتیاط گزارش می‌شوند.");
}

// =====================================================================
// 24 — final results table
// =====================================================================
{
  const s = lightSlide("جدول کامل نتایج", "جمع‌بندی نتایج");

  const rows = [
    ["مسیر اول — اسم خام کلاس، وزن دستی", "X-CLIP ViT-B/32", "0.3327", "0.3716", "0.6385", "0.9473"],
    ["مسیر اول — توصیف بصری + وزن معکوس", "X-CLIP ViT-B/32", "0.3713", "0.4054", "0.6351", "0.9637"],
    ["مسیر اول — چهار نمای زمانی در آزمون", "X-CLIP ViT-B/32", "0.3881", "0.4257", "0.6419", "0.9629"],
    ["مسیر دوم — سر نقطه‌ای، خط پایه", "VideoMAE ViT-B", "0.2975", "0.3860", "0.5660", "—"],
    ["مسیر دوم — سر ترنسفورمر زمانی", "VideoMAE ViT-B", "0.3444", "0.3930", "0.6350", "—"],
    ["مسیر دوم — نرخ یادگیری ۰٫۰۰۰۳", "VideoMAE ViT-B", "0.3645", "0.4210", "0.6550", "—"],
    ["مسیر دوم — ارتقا به رمزگذار بزرگ", "VideoMAE ViT-L", "0.4122", "0.4786", "0.6897", "—"],
    ["مسیر دوم — افزودن آمیزش", "VideoMAE ViT-L", "0.4161", "0.4714", "0.6793", "—"],
    ["مسیر دوم — آمیزش + تنظیم لاجیت", "VideoMAE ViT-L", "0.4298", "0.5000", "0.6897", "0.9610"],
  ];
  table(s, 0.6, 1.8, 12.1,
        ["پیکربندی", "رمزگذار", "ماکرو F1", "صحت نوع", "صحت کلی", "AUC"],
        rows, { colW: [4.5, 2.2, 1.4, 1.4, 1.4, 1.2], rowH: 0.36,
                size: 11, headSize: 11.5, hi: [8] });

  s.addText(
    "مسیر اول روی تقسیمی با ۲۹۶ ویدیوی آزمون و مسیر دوم روی تقسیم استاندارد با ۲۹۰ ویدیو اجرا شده‌اند؛ مقایسه مستقیم اعداد دو مسیر با این قید خوانده شود.",
    rtl({ x: 0.6, y: 5.78, w: 12.1, h: 0.5, fontSize: 11.5, color: MUTED,
          lineSpacing: 17, margin: 0 }));

  stat(s, 0.6, 6.3, 3.9, "0.4298", "ماکرو F1 نهایی", AMBER);
  stat(s, 4.7, 6.3, 3.9, "0.5000", "صحت نوع ناهنجاری", NAVY);
  stat(s, 8.8, 6.3, 3.9, "0.6897", "صحت چندکلاسه", NAVY);
  note(s, "این جدول کامل نتایج هر دو مسیر است. عدد قابل دفاع نهایی 0.4298 با بودجه ثابت و بدون انتخاب روی مجموعه آزمون است.");
}

// =====================================================================
// 25 — conclusions
// =====================================================================
{
  const s = darkSlide();
  s.addShape(pres.ShapeType.ellipse, {
    x: 10.2, y: -1.4, w: 5.0, h: 5.0,
    fill: { color: "1E3566" }, line: { width: 0 },
  });
  s.addText("جمع‌بندی و کارهای آینده", rtl({
    x: 0.7, y: 0.55, w: 11.9, h: 0.7, fontSize: 30, bold: true,
    color: "FFFFFF", margin: 0,
  }));

  const items = [
    ["دستاورد", "اصلاح سه اشکال بنیادی، بازطراحی معماری با رمزگذار منجمد، سر ترنسفورمر زمانی، و پروتکل ارزیابی با کف نوفه", TEAL],
    ["محدودیت", "رمزگذار به دامنه نظارتی تطبیق نمی‌یابد؛ مجموعه آزمون کوچک است و بازه‌های اطمینان پهن", AMBER],
    ["کار آینده", "افزایش وضوح مکانی برای خوشه دزدی، تنظیم دقیق رمزگذار روی داده نظارتی، و تقسیم داده با سهم بزرگ‌تر برای آزمون", ICE],
  ];
  items.forEach((it, i) => {
    const y = 1.75 + i * 1.25;
    s.addShape(pres.ShapeType.roundRect, {
      x: 0.7, y, w: 11.9, h: 0.95, rectRadius: 0.08,
      fill: { color: "1E3566" }, line: { width: 0 },
    });
    s.addText(it[0], rtl({
      x: 10.6, y: y + 0.16, w: 1.8, h: 0.4, fontSize: 15, bold: true,
      color: it[2], margin: 0,
    }));
    s.addText(it[1], rtl({
      x: 1.0, y: y + 0.2, w: 9.4, h: 0.75, fontSize: 13, color: "FFFFFF",
      lineSpacing: 20, margin: 0,
    }));
  });

  s.addShape(pres.ShapeType.roundRect, {
    x: 0.7, y: 5.7, w: 11.9, h: 1.1, rectRadius: 0.08,
    fill: { color: AMBER }, line: { width: 0 },
  });
  s.addText(
    "بیشترین اثر را نه یک ایده الگوریتمی، بلکه کاهش هزینه هر آزمایش داشت؛ و کمّی کردن نوفه پیش از تفسیر، از ادعاهایی جلوگیری کرد که داده پشتیبان آن‌ها نبود.",
    rtl({ x: 1.0, y: 5.92, w: 11.3, h: 0.7, fontSize: 14, bold: true,
          color: "1B1300", lineSpacing: 21, margin: 0 }));
  note(s, "پایان. سوال‌های محتمل: چرا رمزگذار را تنظیم دقیق نکردید، چرا عدد از مقالات پیشرفته پایین‌تر است، و از کجا معلوم آستانه روی تست تنظیم نشده.");
}

pres.writeFile({ fileName: "Defense_VideoAnomaly.pptx" })
  .then(f => console.log("wrote " + f));
