const supported = ["en", "fr", "zh"];
let currentLang = localStorage.getItem("lang") || "en";

const switcher = document.getElementById("lang-switcher");

switcher.addEventListener("click", (e) => {
  if (e.target.tagName === "BUTTON") {
    const lang = e.target.getAttribute("data-lang");
    setActiveLang(lang);
    localStorage.setItem("lang", lang);
    loadLanguage(lang);
  }
});

function setActiveLang(lang) {
  document.querySelectorAll("#lang-switcher button").forEach(btn => {
    btn.classList.toggle("active", btn.getAttribute("data-lang") === lang);
  });
}

function loadLanguage(lang) {
  fetch(`./assets/data/${lang}.json`)
    .then(res => {
      if (!res.ok) throw new Error("JSON not found");
      return res.json();
    })
    .then(data => renderPage(data))
    .catch(err => console.error("Error loading language:", err));
}

function renderPage(data) {
  // Navigation dynamique
  document.getElementById("nav-links").innerHTML = `
    <li><a href="#abstract-section">${data.nav.abstract}</a></li>
    <li><a href="#contributions-section">${data.nav.contributions}</a></li>
    <li><a href="#resources-list">${data.nav.resources}</a></li>
    <li><a href="https://axeldlv00.github.io/">${data.nav.back_home}</a></li>
  `;

  // Hero Section avec image Delaunay
  document.getElementById("hero").innerHTML = `
    <div>
      <h1>${data.hero.title}</h1>
      <p>${data.hero.subtitle}</p>
      <div class="tags">${data.hero.tags.map(t => `<span class="tag">${t}</span>`).join("")}</div>
    </div>
    <a href="https://www.mam.paris.fr/fr/oeuvre/rythme-ndeg1" target="_blank">
      <img src="assets/img/hero.jpg" alt="Delaunay Artwork" class="clickable-hero">
    </a>
  `;

  // Badges (ArXiv, HF, etc.)
  document.getElementById("badges-container").innerHTML = data.badges.map(b => `
    <a class="social-button" href="${b.link}" target="_blank">
      <img src="assets/img/icons/${b.icon}" alt="${b.name}"> ${b.name}
    </a>
  `).join("");

  // Sections de contenu
  buildSection("abstract-section", [data.abstract], data.nav.abstract);
  buildSection("contributions-section", data.contributions, data.nav.contributions);
  buildSection("resources-list", data.resources, data.nav.resources);

  // Citation & License
  document.getElementById("academic-footer").innerHTML = `
    <h2>${data.nav.citation}</h2>
    <div class="card citation-card">
      <pre><code>${data.citation}</code></pre>
    </div>
    <div style="margin-top: 2rem; font-size: 0.9rem; color: var(--muted);">
      <strong>License:</strong> ${data.license}
    </div>
  `;

  // Footer
  document.getElementById("footer-content").innerHTML = `
    © 2025 Axel Delaval — <a href="https://axeldlv00.github.io/">${data.nav.back_home}</a>
  `;
}

function buildSection(id, items, title) {
  const section = document.getElementById(id);
  if (!items) return;

  section.innerHTML = `<h2>${title}</h2>` +
    items.map(item => `
        <div class="card">
          <div class="card-main">
            <div class="card-text">
              <h3>${item.link ? `<a href="${item.link}" target="_blank">${item.title}</a>` : item.title}</h3>
              ${item.meta ? `<p class="meta">${item.meta}</p>` : ""}
              <p class="desc">${item.desc}</p>
            </div>
            ${item.logo ? `
                <div class="visual-column">
                  <img class="inline-logo" src="assets/img/icons/${item.logo}" alt="logo">
                </div>
              ` : ""}
          </div>
        </div>
      `).join("");
}

// Initialisation
setActiveLang(currentLang);
loadLanguage(currentLang);