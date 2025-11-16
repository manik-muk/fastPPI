document.addEventListener("DOMContentLoaded", () => {
  const yearSpan = document.getElementById("year");
  if (yearSpan) {
    yearSpan.textContent = new Date().getFullYear().toString();
  }

  const form = document.getElementById("waitlist-form");
  if (!form) return;

  // --- Formspree integration ------------------------------------------------
  // Formspree is a free service that handles form submissions for static sites.
  // Sign up at https://formspree.io/ to get your form endpoint.
  // Then replace YOUR_FORM_ID below with your actual Formspree form ID.

  const FORMSPREE_ENDPOINT = "https://formspree.io/f/mvgdvdgl";

  const submitButton = form.querySelector('button[type="submit"]');

  form.addEventListener("submit", async (e) => {
    e.preventDefault();

    const formData = new FormData(form);

    if (submitButton) {
      submitButton.disabled = true;
      submitButton.textContent = "Submitting…";
    }

    try {
      const response = await fetch(FORMSPREE_ENDPOINT, {
        method: "POST",
        body: formData,
        headers: {
          Accept: "application/json",
        },
      });

      if (response.ok) {
        if (submitButton) {
          submitButton.disabled = true;
          submitButton.textContent = "You're on the list";
        }

        const note = document.createElement("p");
        note.className = "waitlist-footnote";
        note.textContent =
          "Thanks for joining the waitlist — we'll reach out as fastPPI is ready for more users.";
        form.appendChild(note);

        form.reset();
      } else {
        throw new Error("Form submission failed");
      }
    } catch (error) {
      if (submitButton) {
        submitButton.disabled = false;
        submitButton.textContent = "Join waitlist";
      }
      alert("Oops! There was a problem submitting the form. Please try again.");
    }
  });
});

