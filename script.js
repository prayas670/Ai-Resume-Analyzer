const API_BASE = ""; // same-origin; change to e.g. "http://localhost:5000" if serving frontend separately

const dropzone = document.getElementById("dropzone");
const fileInput = document.getElementById("fileInput");
const dzTitle = document.getElementById("dzTitle");
const dzSub = document.getElementById("dzSub");
const scanBeam = document.getElementById("scanBeam");
const fileChip = document.getElementById("fileChip");
const fileNameEl = document.getElementById("fileName");
const clearFileBtn = document.getElementById("clearFile");
const analyzeBtn = document.getElementById("analyzeBtn");
const jdInput = document.getElementById("jdInput");
const aiToggle = document.getElementById("aiToggle");
const errorMsg = document.getElementById("errorMsg");
const report = document.getElementById("report");
const downloadPdfBtn = document.getElementById("downloadPdfBtn");

let selectedFile = null;

// --- Dropzone interactions -------------------------------------------------

dropzone.addEventListener("click", () => fileInput.click());
dropzone.addEventListener("keydown", (e) => {
  if (e.key === "Enter" || e.key === " ") { e.preventDefault(); fileInput.click(); }
});

["dragenter", "dragover"].forEach((evt) =>
  dropzone.addEventListener(evt, (e) => {
    e.preventDefault();
    dropzone.classList.add("dragover");
  })
);
["dragleave", "drop"].forEach((evt) =>
  dropzone.addEventListener(evt, (e) => {
    e.preventDefault();
    dropzone.classList.remove("dragover");
  })
);
dropzone.addEventListener("drop", (e) => {
  const file = e.dataTransfer.files[0];
  if (file) setFile(file);
});

fileInput.addEventListener("change", () => {
  if (fileInput.files[0]) setFile(fileInput.files[0]);
});

clearFileBtn.addEventListener("click", (e) => {
  e.stopPropagation();
  selectedFile = null;
  fileInput.value = "";
  fileChip.hidden = true;
  dropzone.classList.remove("has-file");
  dzTitle.textContent = "Drop resume here";
  dzSub.textContent = "PDF, DOCX or TXT — max 5MB";
  analyzeBtn.disabled = true;
});

function setFile(file) {
  const ext = file.name.split(".").pop().toLowerCase();
  if (!["pdf", "docx", "txt"].includes(ext)) {
    showError("Unsupported file type. Please upload a PDF, DOCX, or TXT file.");
    return;
  }
  if (file.size > 5 * 1024 * 1024) {
    showError("File too large — max 5MB.");
    return;
  }
  hideError();
  selectedFile = file;
  fileNameEl.textContent = file.name;
  fileChip.hidden = false;
  dropzone.classList.add("has-file");
  dzTitle.textContent = "Ready to scan";
  dzSub.textContent = "Click 'Run scan' to analyze";
  analyzeBtn.disabled = false;
}

function showError(msg) {
  errorMsg.textContent = msg;
  errorMsg.hidden = false;
}
function hideError() {
  errorMsg.hidden = true;
}

// --- Analyze ----------------------------------------------------------------

analyzeBtn.addEventListener("click", async () => {
  if (!selectedFile) return;
  hideError();
  report.hidden = true;
  setScanning(true);

  const formData = new FormData();
  formData.append("resume", selectedFile);
  formData.append("job_description", jdInput.value.trim());
  formData.append("ai_feedback", aiToggle.checked ? "true" : "false");

  try {
    const res = await fetch(`${API_BASE}/api/analyze`, {
      method: "POST",
      body: formData,
    });
    const data = await res.json();
    if (!res.ok) {
      showError(data.error || "Something went wrong analyzing your resume.");
      return;
    }
    renderReport(data);
  } catch (err) {
    showError("Could not reach the analysis server. Is the backend running?");
  } finally {
    setScanning(false);
  }
});

function setScanning(on) {
  analyzeBtn.disabled = on || !selectedFile;
  analyzeBtn.querySelector(".btn-label").textContent = on ? "Scanning..." : "Run scan";
  scanBeam.classList.toggle("scanning", on);
}

// --- Rendering ----------------------------------------------------------------

function renderReport(data) {
  report.hidden = false;

  animateGauge(data.overall_score);

  const levelBadge = document.getElementById("candidateLevelBadge");
  if (levelBadge && data.candidate_level) {
    levelBadge.textContent = data.candidate_level + " Profile";
  }

  const domainBadge = document.getElementById("domainBadge");
  if (domainBadge && data.domain && data.domain.domain) {
    domainBadge.textContent = data.domain.domain;
  }

  const experienceBadge = document.getElementById("experienceBadge");
  if (experienceBadge && data.experience_duration) {
    const yrs = data.experience_duration.years;
    experienceBadge.textContent = yrs > 0 ? yrs + " YOE" : "No Exp Detected";
  }

  const subRow = document.getElementById("subscoreRow");
  subRow.innerHTML = "";
  const subs = [
    ["Structure", data.structure_score],
    ["Content quality", data.content_score],
    ["Skills coverage", data.skills_score],
  ];
  if (data.jd_match) subs.push(["JD similarity", data.jd_match.similarity]);
  subs.forEach(([label, val]) => {
    const div = document.createElement("div");
    div.className = "subscore";
    div.innerHTML = `<span class="subscore-label">${label}</span><span class="subscore-value">${val}</span>`;
    subRow.appendChild(div);
  });

  renderSectionScores(data.section_scores);
  renderHeatmap(data.section_scores);
  renderCompleteness(data.completeness);
  renderAtsRisk(data.ats_risk);
  renderBulletRewrites(data.bullet_rewrites);
  renderStarAnalysis(data.star_analysis);
  renderProjectEnhancements(data.project_enhancements);
  renderDashboard(data);

  const checklist = document.getElementById("checklist");
  checklist.innerHTML = "";
  data.structure_checks.forEach(([label, passed]) => {
    const li = document.createElement("li");
    li.className = passed ? "pass" : "fail";
    li.innerHTML = `<span class="mark">${passed ? "✓" : "✕"}</span><span>${label}</span>`;
    checklist.appendChild(li);
  });

  const skillChips = document.getElementById("skillChips");
  skillChips.innerHTML = "";
  if (data.skills_found.length === 0) {
    skillChips.innerHTML = `<span class="dz-sub">No known skills detected — consider adding a Skills section.</span>`;
  } else {
    data.skills_found.forEach((s) => {
      const span = document.createElement("span");
      span.className = "chip";
      span.textContent = s;
      skillChips.appendChild(span);
    });
  }

  renderJDMatch(data);

  const suggList = document.getElementById("suggestionsList");
  suggList.innerHTML = "";
  data.suggestions.forEach((s) => {
    const li = document.createElement("li");
    li.textContent = s;
    suggList.appendChild(li);
  });

  const aiCard = document.getElementById("aiCard");
  if (data.ai_feedback) {
    aiCard.hidden = false;
    document.getElementById("aiFeedback").textContent = data.ai_feedback;
  } else {
    aiCard.hidden = true;
  }

  report.scrollIntoView({ behavior: "smooth", block: "start" });
}

function renderKeywordChips(containerId, keywords, chipClass, emptyText) {
  const el = document.getElementById(containerId);
  el.innerHTML = "";
  if (!keywords || !keywords.length) {
    const span = document.createElement("span");
    span.className = "dz-sub";
    span.textContent = emptyText;
    el.appendChild(span);
    return;
  }
  keywords.forEach((k) => {
    const span = document.createElement("span");
    span.className = `chip ${chipClass}`;
    span.textContent = k;
    el.appendChild(span);
  });
}

function renderDensityChips(containerId, densityMap) {
  const el = document.getElementById(containerId);
  el.innerHTML = "";
  const entries = Object.entries(densityMap);
  if (entries.length === 0) {
    const span = document.createElement("span");
    span.className = "dz-sub";
    span.textContent = "None found";
    el.appendChild(span);
    return;
  }
  entries.sort((a, b) => b[1] - a[1]); // Sort by density descending
  entries.forEach(([k, count]) => {
    const span = document.createElement("span");
    span.className = `chip match density`;
    span.innerHTML = `${escapeHtml(k)} <span class="density-badge">${count}</span>`;
    el.appendChild(span);
  });
}

function scoreColor(score) {
  if (score < 50) return "#E85D4F"; // coral
  if (score < 75) return "#FFB627"; // amber
  return "#2E9E6B"; // green
}

// --- Section-wise score ------------------------------------------------------

function renderSectionScores(sectionScores) {
  const wrap = document.getElementById("sectionBars");
  wrap.innerHTML = "";
  if (!sectionScores) return;

  Object.values(sectionScores).forEach((s) => {
    const row = document.createElement("div");
    row.className = "section-bar-row" + (s.present ? "" : " not-present");

    const label = document.createElement("div");
    label.className = "section-bar-label";
    let subtext = "";
    if (!s.present) subtext = "Not found";
    else if (typeof s.count === "number") subtext = `${s.count} skills matched`;
    else if (typeof s.bullets_detected === "number") subtext = `${s.bullets_detected} bullets detected`;
    label.innerHTML = `${s.label}${subtext ? `<span class="sub">${subtext}</span>` : ""}`;

    const track = document.createElement("div");
    track.className = "section-bar-track";
    const fill = document.createElement("div");
    fill.className = "section-bar-fill";
    const val = s.score == null ? 0 : s.score;
    fill.style.width = "0%";
    fill.style.background = s.present ? scoreColor(val) : "var(--line)";
    track.appendChild(fill);

    const valueEl = document.createElement("div");
    valueEl.className = "section-bar-value";
    valueEl.textContent = s.score == null ? "—" : `${s.score}`;

    row.appendChild(label);
    row.appendChild(track);
    row.appendChild(valueEl);
    wrap.appendChild(row);

    requestAnimationFrame(() => { fill.style.width = `${val}%`; });
  });
}
// --- Resume Heatmap ----------------------------------------------------------

function renderHeatmap(sectionScores) {
  const container = document.getElementById("heatmapContainer");
  container.innerHTML = "";
  if (!sectionScores) return;

  Object.values(sectionScores).forEach(s => {
    if (!s.present) return;
    const block = document.createElement("div");
    block.className = "heatmap-block";
    block.style.backgroundColor = scoreColor(s.score || 0);
    
    let weight = 1;
    if (s.label.toLowerCase().includes("experience")) weight = 3;
    if (s.label.toLowerCase().includes("education")) weight = 1.5;
    if (s.label.toLowerCase().includes("skills")) weight = 2;
    if (s.label.toLowerCase().includes("projects")) weight = 2;
    
    block.style.flexGrow = weight;
    
    // Add text color contrast if needed, scoreColor returns dark/light backgrounds
    // Coral/Green/Amber are all relatively dark, so white text usually works best
    block.style.color = "#fff";
    
    block.innerHTML = `<span class="heatmap-label">${escapeHtml(s.label)}</span><span class="heatmap-score">${s.score}</span>`;
    container.appendChild(block);
  });
}
// --- Completeness score --------------------------------------------------------

function renderCompleteness(completeness) {
  if (!completeness) return;
  document.getElementById("completenessPct").textContent = `${completeness.score}%`;

  const fill = document.getElementById("completenessBarFill");
  fill.style.width = "0%";
  requestAnimationFrame(() => { fill.style.width = `${completeness.score}%`; });

  const list = document.getElementById("completenessChecklist");
  list.innerHTML = "";
  completeness.checklist.forEach(([label, passed]) => {
    const li = document.createElement("li");
    li.className = passed ? "pass" : "fail";
    li.innerHTML = `<span class="mark">${passed ? "✓" : "✕"}</span><span>${label}</span>`;
    list.appendChild(li);
  });
}

// --- ATS risk analysis --------------------------------------------------------

function renderAtsRisk(atsRisk) {
  if (!atsRisk) return;
  const badge = document.getElementById("riskBadge");
  badge.textContent = `${atsRisk.risk_level} risk`;
  badge.className = `risk-badge ${atsRisk.risk_level.toLowerCase()}`;

  const list = document.getElementById("atsIssues");
  list.innerHTML = "";
  if (!atsRisk.issues.length) {
    const p = document.createElement("p");
    p.className = "ats-clean";
    p.textContent = "No common ATS red flags detected — tables, images, multi-column layouts, and non-standard headers all look clear.";
    list.appendChild(p);
    return;
  }
  atsRisk.issues.forEach((issue) => {
    const li = document.createElement("li");
    li.className = issue.severity;
    li.innerHTML = `<span class="ats-issue-title">${issue.issue}</span><p class="ats-issue-tip">${issue.tip}</p>`;
    list.appendChild(li);
  });
}

// --- Bullet point rewrites --------------------------------------------------------

function renderBulletRewrites(bulletRewrites) {
  const card = document.getElementById("bulletCard");
  const list = document.getElementById("bulletList");
  const subtitle = document.getElementById("bulletSubtitle");
  list.innerHTML = "";

  if (!bulletRewrites || bulletRewrites.length === 0) {
    card.hidden = true;
    return;
  }
  card.hidden = false;
  const aiUsed = bulletRewrites.some((b) => b.ai_powered);
  subtitle.textContent = aiUsed
    ? "Rewrites for your weakest bullet points, generated by the AI reviewer."
    : "Rewrites for your weakest bullet points, using strong action verbs and quantified impact.";

  bulletRewrites.forEach((b) => {
    const item = document.createElement("div");
    item.className = "bullet-item";
    item.innerHTML = `
      <div class="bullet-original"><span class="bullet-tag before">Before</span><span>${escapeHtml(b.original)}</span></div>
      <div class="bullet-suggested"><span class="bullet-tag after">After</span><span>${escapeHtml(b.suggested)}${b.ai_powered ? '<span class="ai-pill">AI</span>' : ""}</span></div>
      ${b.notes && b.notes.length ? `<ul class="bullet-notes">${b.notes.map((n) => `<li>${escapeHtml(n)}</li>`).join("")}</ul>` : ""}
    `;
    list.appendChild(item);
  });
}

function escapeHtml(str) {
  const div = document.createElement("div");
  div.textContent = str;
  return div.innerHTML;
}

// --- STAR Method Analysis -----------------------------------------------------

function renderStarAnalysis(starAnalysis) {
  const card = document.getElementById("starCard");
  const list = document.getElementById("starList");
  list.innerHTML = "";
  
  if (!starAnalysis || starAnalysis.length === 0) {
    card.hidden = true;
    return;
  }
  card.hidden = false;
  
  starAnalysis.forEach(item => {
    const div = document.createElement("div");
    div.className = "bullet-item";
    
    let badges = [];
    if (item.has_action) badges.push(`<span class="star-badge action">Action</span>`);
    if (item.has_context) badges.push(`<span class="star-badge context">Context</span>`);
    if (item.has_result) badges.push(`<span class="star-badge result">Result</span>`);
    
    if (badges.length === 0) {
       badges.push(`<span class="star-badge none">Needs Improvement</span>`);
    }
    
    div.innerHTML = `
      <div class="bullet-original" style="margin-bottom: 8px;"><span>${escapeHtml(item.text)}</span></div>
      <div class="star-badges">${badges.join("")}</div>
    `;
    list.appendChild(div);
  });
}

function animateGauge(score) {
  const circumference = 377; // 2 * PI * 60, matches stroke-dasharray in CSS
  const fill = document.getElementById("gaugeFill");
  const offset = circumference - (score / 100) * circumference;

  let color = "#2E9E6B"; // green
  if (score < 50) color = "#E85D4F"; // coral
  else if (score < 75) color = "#FFB627"; // amber
  fill.style.stroke = color;

  // reset then animate
  fill.style.transition = "none";
  fill.style.strokeDashoffset = circumference;
  requestAnimationFrame(() => {
    fill.style.transition = "stroke-dashoffset 1s cubic-bezier(0.65,0,0.35,1), stroke 0.4s ease";
    fill.style.strokeDashoffset = offset;
  });

  let current = 0;
  const target = score;
  const duration = 800;
  const start = performance.now();
  const numberEl = document.getElementById("scoreNumber");
  function tick(now) {
    const progress = Math.min((now - start) / duration, 1);
    current = Math.round(progress * target);
    numberEl.textContent = current;
    if (progress < 1) requestAnimationFrame(tick);
  }
  requestAnimationFrame(tick);
}

// --- Dashboard & PDF Export & Enhancements --------------------------------------

if (downloadPdfBtn) {
  downloadPdfBtn.addEventListener("click", () => {
    const element = document.getElementById("report");
    downloadPdfBtn.style.display = "none";
    html2pdf().from(element).set({
      margin: 0.5,
      filename: 'Resume-Analysis-Report.pdf',
      jsPDF: { unit: 'in', format: 'letter', orientation: 'portrait' }
    }).save().then(() => {
      downloadPdfBtn.style.display = "inline-block";
    });
  });
}

let radarChartInstance = null;
let barChartInstance = null;

function renderDashboard(data) {
  document.getElementById("dashboardGrid").style.display = "grid";

  const ctxRadar = document.getElementById("radarChart").getContext("2d");
  if (radarChartInstance) radarChartInstance.destroy();

  const labelsRadar = ["Structure", "Content", "Skills"];
  const dataRadar = [data.structure_score, data.content_score, data.skills_score];
  if (data.jd_match) {
    labelsRadar.push("JD Match");
    dataRadar.push(data.jd_match.similarity);
  }

  radarChartInstance = new Chart(ctxRadar, {
    type: 'radar',
    data: {
      labels: labelsRadar,
      datasets: [{
        label: 'Score',
        data: dataRadar,
        backgroundColor: 'rgba(46, 158, 107, 0.2)',
        borderColor: '#2E9E6B',
        pointBackgroundColor: '#2E9E6B',
      }]
    },
    options: {
      responsive: true,
      maintainAspectRatio: false,
      scales: {
        r: { min: 0, max: 100 }
      }
    }
  });

  const ctxBar = document.getElementById("barChart").getContext("2d");
  if (barChartInstance) barChartInstance.destroy();

  const labelsBar = [];
  const dataBar = [];
  Object.values(data.section_scores).forEach(s => {
    labelsBar.push(s.label);
    dataBar.push(s.score || 0);
  });

  barChartInstance = new Chart(ctxBar, {
    type: 'bar',
    data: {
      labels: labelsBar,
      datasets: [{
        label: 'Section Score',
        data: dataBar,
        backgroundColor: '#FFB627',
      }]
    },
    options: {
      responsive: true,
      maintainAspectRatio: false,
      scales: {
        y: { beginAtZero: true, max: 100 }
      }
    }
  });
}

let skillGapChartInstance = null;
function renderSkillGap(matchedList, missingList) {
  const ctx = document.getElementById("skillGapChart").getContext("2d");
  if (skillGapChartInstance) skillGapChartInstance.destroy();
  
  const matchedCount = matchedList ? matchedList.length : 0;
  const missingCount = missingList ? missingList.length : 0;
  
  skillGapChartInstance = new Chart(ctx, {
    type: 'doughnut',
    data: {
      labels: ['Matched', 'Missing'],
      datasets: [{
        data: [matchedCount, missingCount],
        backgroundColor: ['#2E9E6B', '#E85D4F'],
        borderWidth: 0
      }]
    },
    options: {
      responsive: true,
      maintainAspectRatio: false,
      plugins: {
        legend: { position: 'bottom' }
      }
    }
  });
}

function renderProjectEnhancements(projectEnhancements) {
  const card = document.getElementById("projectCard");
  const list = document.getElementById("projectList");
  const subtitle = document.getElementById("projectSubtitle");
  list.innerHTML = "";

  if (!projectEnhancements || projectEnhancements.length === 0) {
    card.hidden = true;
    return;
  }
  card.hidden = false;
  const aiUsed = projectEnhancements.some((b) => b.ai_powered);
  subtitle.textContent = aiUsed
    ? "Enhancements for your project descriptions, generated by the AI reviewer."
    : "Enhancements for your project descriptions, focusing on metrics and technologies.";

  projectEnhancements.forEach((b) => {
    const item = document.createElement("div");
    item.className = "bullet-item";
    item.innerHTML = `
      <div class="bullet-original"><span class="bullet-tag before">Original</span><span>${escapeHtml(b.original)}</span></div>
      <div class="bullet-suggested"><span class="bullet-tag after">Suggested</span><span>${escapeHtml(b.suggested)}${b.ai_powered ? '<span class="ai-pill">AI</span>' : ""}</span></div>
      ${b.notes && b.notes.length ? '<ul class="bullet-notes">' + b.notes.map(n => '<li>' + escapeHtml(n) + '</li>').join("") + '</ul>' : ""}
    `;
    list.appendChild(item);
  });
}

function renderProjectQuality(data) {
  const c = document.getElementById("projectQualityCard");
  if (!data.project_quality || !data.project_quality.metrics) {
    c.hidden = true;
    return;
  }
  c.hidden = false;

  const r = document.getElementById("projectSubscoreRow");
  r.innerHTML = "";

  const pq = data.project_quality;
  const subs = [
    { label: "Overall", score: pq.score },
    { label: "Tech Stack", score: pq.metrics.tech_stack_score },
    { label: "Complexity", score: pq.metrics.complexity_score },
    { label: "Impact", score: pq.metrics.impact_score },
    { label: "Action Verbs", score: pq.metrics.action_verbs_score }
  ];

  subs.forEach(s => {
    const div = document.createElement("div");
    div.className = "subscore";
    div.innerHTML = `<span class="subscore-label">${s.label}</span><span class="subscore-value">${s.score}</span>`;
    r.appendChild(div);
  });

  const sl = document.getElementById("projectSuggestionsList");
  sl.innerHTML = "";
  pq.suggestions.forEach(sug => {
    const li = document.createElement("li");
    li.textContent = sug;
    sl.appendChild(li);
  });
}

function renderJDMatch(data) {
  const c = document.getElementById("jdCard");
  if (!data.jd_match) {
    c.hidden = true;
    return;
  }
  c.hidden = false;

  const jd = data.jd_match;
  document.getElementById("jdSimilarity").textContent = jd.similarity + "%";

  renderDensityChips("reqMatchedChips", jd.required_matched || [], jd.keyword_density, true);
  renderDensityChips("reqMissingChips", jd.required_missing || [], jd.keyword_density, false);
  
  const prefWrap = document.getElementById("preferredSkillsWrap");
  if ((jd.preferred_matched && jd.preferred_matched.length > 0) || (jd.preferred_missing && jd.preferred_missing.length > 0)) {
      prefWrap.hidden = false;
      renderDensityChips("prefMatchedChips", jd.preferred_matched || [], jd.keyword_density, true);
      renderDensityChips("prefMissingChips", jd.preferred_missing || [], jd.keyword_density, false);
  } else {
      prefWrap.hidden = true;
  }
  
  const reqsSection = document.getElementById("jdReqsSection");
  if (jd.education_requirements || jd.experience_requirements) {
      reqsSection.hidden = false;
      document.getElementById("jdEdu").textContent = jd.education_requirements || "Not specified";
      document.getElementById("jdExp").textContent = jd.experience_requirements || "Not specified";
  } else {
      reqsSection.hidden = true;
  }

  const allMatched = (jd.required_matched || []).concat(jd.preferred_matched || []);
  const allMissing = (jd.required_missing || []).concat(jd.preferred_missing || []);
  renderSkillGap(allMatched, allMissing);
}
