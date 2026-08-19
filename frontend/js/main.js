// ============================================
// Customer Behavior Prediction Platform
// Frontend Logic (Firebase static mode)
// ============================================

const DATA_BASE = "./data";

async function loadOverviewMetrics() {
    const totalEl = document.getElementById("total-customers");
    const churnEl = document.getElementById("churn-rate");
    const lowEl = document.getElementById("low-engagement");
    const avgEl = document.getElementById("avg-value");

    if (!totalEl && !churnEl && !lowEl && !avgEl) return;

    try {
        const response = await fetch(`${DATA_BASE}/metrics.json`);
        if (!response.ok) throw new Error("Failed to fetch metrics");
        const data = await response.json();

        if (totalEl) totalEl.textContent = data.total_customers.toLocaleString();
        if (churnEl) churnEl.textContent = data.churn_rate + "%";
        if (lowEl) lowEl.textContent = data.low_engagement.toLocaleString();
        if (avgEl) avgEl.textContent = "£" + data.avg_customer_value.toLocaleString();
    } catch (error) {
        console.error("Error loading metrics:", error);
    }
}

function setupMobileMenu() {
    const sidebar = document.getElementById("sidebar");
    const toggle = document.getElementById("menuToggle");
    if (sidebar && toggle) {
        toggle.addEventListener("click", () => {
            sidebar.classList.toggle("open");
        });
    }
}

document.addEventListener("DOMContentLoaded", () => {
    setupMobileMenu();
    loadOverviewMetrics();
});