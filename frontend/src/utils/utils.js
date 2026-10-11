const usd0 = new Intl.NumberFormat("en-US", {
  style: "currency",
  currency: "USD",
  maximumFractionDigits: 0,
});

const usd2 = new Intl.NumberFormat("en-US", {
  style: "currency",
  currency: "USD",
});

// Format whole prices as $35 and decimal prices as $24.99.
export function formatPrice(n) {
  if (n === null || n === undefined) {
    return "Price unavailable";
  }

  return Number.isInteger(n) ? usd0.format(n) : usd2.format(n);
}

export function formatRating(rating, count) {
  if (rating === null || rating === undefined) {
    return "No ratings yet";
  }

  const reviews =
    count === null || count === undefined
      ? ""
      : ` · ${count.toLocaleString("en-US")} reviews`;

  return `★ ${rating.toFixed(1)}${reviews}`;
}

// Prepare recent messages for the backend API.
// Failed turns are skipped.
export function buildHistory(messages, max = 6) {
  return messages
    .filter((m) => !m.error && m.content)
    .slice(-max)
    .map((m) => ({
      role: m.role,
      content: m.content,
    }));
}
