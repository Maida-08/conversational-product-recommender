function formatPrice(price) {
  if (price === null || price === undefined) return "Price unavailable";
  return new Intl.NumberFormat("en-US", {
    style: "currency",
    currency: "USD",
  }).format(price);
}

export default function ProductCard({ product }) {
  if (!product) return null;

  return (
    <article className={`product-card product-card--${product.role || "primary"}`}>
      <div className="product-card__top">
        <span className="product-card__role">{product.role || "Product"}</span>
        <span className="product-card__match">
          {product.match_label || "Match"}
        </span>
      </div>

      <h3>{product.title}</h3>
      <p className="product-card__brand">
        {product.brand} · {product.category}
      </p>

      <div className="product-card__price">{formatPrice(product.price_usd)}</div>

      {product.average_rating != null && (
        <p className="product-card__rating">
          ★ {product.average_rating.toFixed(1)}
          {product.num_reviews != null && ` (${product.num_reviews} reviews)`}
        </p>
      )}

      {product.reasons?.length > 0 && (
        <ul>
          {product.reasons.map((reason, index) => (
            <li key={`${product.product_id}-reason-${index}`}>{reason}</li>
          ))}
        </ul>
      )}

      {product.warnings?.length > 0 && (
        <div className="product-card__warnings">
          <strong>Things to consider</strong>
          <ul>
            {product.warnings.map((warning, index) => (
              <li key={`${product.product_id}-warning-${index}`}>{warning}</li>
            ))}
          </ul>
        </div>
      )}
    </article>
  );
}
