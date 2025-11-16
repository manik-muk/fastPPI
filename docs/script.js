document.addEventListener("DOMContentLoaded", () => {
  const yearSpan = document.getElementById("year");
  if (yearSpan) {
    yearSpan.textContent = new Date().getFullYear().toString();
  }

  const form = document.getElementById("waitlist-form");
  if (!form) return;

  form.addEventListener("submit", (e) => {
    e.preventDefault();

    const formData = new FormData(form);
    const email = (formData.get("email") || "").toString().trim();
    const company = (formData.get("company") || "").toString().trim();
    const context = (formData.get("context") || "").toString().trim();

    const subject = encodeURIComponent("fastPPI waitlist");
    const bodyLines = [
      "Hi, I'd like to join the fastPPI waitlist.",
      "",
      `Email: ${email || "(not provided)"}`,
      `Company / team: ${company || "(not provided)"}`,
      "",
      "How we preprocess data today:",
      context || "(no additional context provided)",
    ];

    const body = encodeURIComponent(bodyLines.join("\n"));

    // TODO: replace with your preferred contact email
    const recipient = "FOUNDERS_EMAIL@yourdomain.com";
    const mailto = `mailto:${recipient}?subject=${subject}&body=${body}`;

    window.location.href = mailto;
  });
});


