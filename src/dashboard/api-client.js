/**
 * api-client.js
 * -------------
 * Thin JavaScript client for the PropSense FastAPI backend.
 * Include this file in index.html when running with the live API.
 *
 * Usage (in index.html):
 *   <script src="api-client.js"></script>
 *   const result = await ApiClient.predict({ city:"Pune", ... });
 */

const API_BASE = window.API_BASE || "http://localhost:8000";

const ApiClient = (() => {

  /**
   * Generic fetch wrapper with JSON body.
   * @param {string} endpoint
   * @param {object} body
   * @returns {Promise<object>}
   */
  async function post(endpoint, body) {
    const resp = await fetch(`${API_BASE}${endpoint}`, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify(body),
    });
    if (!resp.ok) {
      const err = await resp.json().catch(() => ({ detail: resp.statusText }));
      throw new Error(err.detail || `HTTP ${resp.status}`);
    }
    return resp.json();
  }

  async function get(endpoint) {
    const resp = await fetch(`${API_BASE}${endpoint}`);
    if (!resp.ok) throw new Error(`HTTP ${resp.status}`);
    return resp.json();
  }

  return {
    /**
     * Health check
     */
    health: () => get("/health"),

    /**
     * Predict property price.
     * @param {object} property  - PropertyInput schema fields
     */
    predict: (property) => post("/predict", property),

    /**
     * Get similar property recommendations.
     * @param {object} property       - query property
     * @param {number} topN           - number of results
     * @param {number} priceTolerance - % tolerance on price filter
     * @param {boolean} sameCity      - restrict to same city
     */
    recommend: (property, topN = 5, priceTolerance = 30, sameCity = true) =>
      post("/recommend", {
        property,
        top_n: topN,
        price_tolerance_pct: priceTolerance,
        same_city: sameCity,
      }),

    /**
     * Investment analysis.
     * @param {number} price
     * @param {string} city
     * @param {number} [areaSqft]
     * @param {number} [bedrooms]
     */
    investment: (price, city, areaSqft, bedrooms) =>
      post("/investment", { price, city, area_sqft: areaSqft, bedrooms }),

    /**
     * Anomaly detection.
     * @param {object[]} properties  - array of property objects with 'price'
     */
    anomaly: (properties) => post("/anomaly", { properties }),

    /**
     * List supported cities.
     */
    cities: () => get("/cities"),

    /**
     * Active model info.
     */
    modelInfo: () => get("/model-info"),
  };
})();

/* ── Optional: auto-connect indicator ────────────────────────────── */
(async () => {
  try {
    const h = await ApiClient.health();
    console.log("[API] Connected:", h);
    const el = document.getElementById("api-status");
    if (el) {
      el.textContent = "● API Connected";
      el.style.color = "var(--accent2, #00d4a8)";
    }
  } catch (e) {
    console.warn("[API] Running in offline/demo mode:", e.message);
    const el = document.getElementById("api-status");
    if (el) {
      el.textContent = "● Demo Mode (API offline)";
      el.style.color = "var(--muted, #6b6b80)";
    }
  }
})();
