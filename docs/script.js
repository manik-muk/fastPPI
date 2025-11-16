document.addEventListener("DOMContentLoaded", () => {
  const yearSpan = document.getElementById("year");
  if (yearSpan) {
    yearSpan.textContent = new Date().getFullYear().toString();
  }

  const form = document.getElementById("waitlist-form");
  if (!form) return;

  // --- Google Forms wiring --------------------------------------------------
  //
  // 1. Create a Google Form with fields like:
  //    - Work email (Short answer)
  //    - Company / team (Short answer)
  //    - How are you preprocessing data today? (Paragraph)
  // 2. In the live form page, inspect the HTML and copy the "name" attributes
  //    for each input/textarea (they look like entry.1234567890).
  // 3. Paste them into the constants below, and set GOOGLE_FORM_ACTION
  //    to your form's formResponse URL.
  //
  // Example formResponse URL:
  //   https://docs.google.com/forms/d/e/YOUR_FORM_ID/formResponse

  const GOOGLE_FORM_ACTION =
    "https://docs.google.com/forms/d/e/1FAIpQLScal7ViypEBOrES9MIC2j94y-nER76JK6IkRMpOMSIyxl6uXQ/viewform?usp=header"; // TODO: replace with your formResponse URL

  const FIELD_EMAIL = "entry.Email"; // TODO: replace with your email entry.* name
  const FIELD_COMPANY = "entry.Company"; // TODO: replace with your company entry.* name
  const FIELD_CONTEXT = "entry.Context"; // TODO: replace with your context entry.* name

  const submitButton = form.querySelector('button[type="submit"]');
  let hasConfiguredForm =
    GOOGLE_FORM_ACTION.includes("YOUR_FORM_ID") === false &&
    !FIELD_EMAIL.includes("ENTRY_ID") &&
    !FIELD_COMPANY.includes("ENTRY_ID") &&
    !FIELD_CONTEXT.includes("ENTRY_ID");

  form.addEventListener("submit", (e) => {
    e.preventDefault();

    const formData = new FormData(form);
    const email = (formData.get("email") || "").toString().trim();
    const company = (formData.get("company") || "").toString().trim();
    const context = (formData.get("context") || "").toString().trim();

    if (!hasConfiguredForm) {
      // Fallback: if config isn't filled in yet, just open the Google Form URL
      // (you can replace this with your live form link even before wiring entry IDs).
      const fallbackUrl = GOOGLE_FORM_ACTION.replace("/formResponse", "/viewform");
      window.open(fallbackUrl, "_blank");
      return;
    }

    const googleParams = new URLSearchParams();
    googleParams.append(FIELD_EMAIL, email);
    googleParams.append(FIELD_COMPANY, company);
    googleParams.append(FIELD_CONTEXT, context);

    if (submitButton) {
      submitButton.disabled = true;
      submitButton.textContent = "Submitting…";
    }

    // Use no-cors so the browser doesn't block the POST; we don't need the response.
    fetch(GOOGLE_FORM_ACTION, {
      method: "POST",
      mode: "no-cors",
      body: googleParams,
    })
      .catch(() => {
        // Silent failure – worst case, the form doesn't go through.
      })
      .finally(() => {
        if (submitButton) {
          submitButton.disabled = true;
          submitButton.textContent = "You're on the list";
        }

        const note = document.createElement("p");
        note.className = "waitlist-footnote";
        note.textContent =
          "Thanks for joining the waitlist — we’ll reach out as fastPPI is ready for more users.";
        form.appendChild(note);

        form.reset();
      });
  });
});

