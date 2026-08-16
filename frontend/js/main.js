// ============================================
// Customer Behavior Prediction Platform
// Frontend Logic
// ============================================

const API_URL = "http://127.0.0.1:8000";

/**
 * Fetch overview metrics from the API and update the page
 */
async function loadOverviewMetrics() {
    try {
        const response = await fetch(`${API_URL}/metrics`);
        if (!response.ok) throw new Error("Failed to fetch metrics");

        const data = await response.json();

        // Update the DOM with real values
        document.getElementById("total-customers").textContent =
            data.total_customers.toLocaleString();

        document.getElementById("churn-rate").textContent =
            data.churn_rate + "%";

        document.getElementById("low-engagement").textContent =
            data.low_engagement.toLocaleString();

        document.getElementById("avg-value").textContent =
            "£" + data.avg_customer_value.toLocaleString();

    } catch (error) {
        console.error("Error loading metrics:", error);
        // Keep the fallback values if API is not available
    }
}

// Run when the page loads
document.addEventListener("DOMContentLoaded", function () {
    // Only run on the overview page
    if (document.getElementById("total-customers")) {
        loadOverviewMetrics();
    }
});