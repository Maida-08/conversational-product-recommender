import ProductCard from "./ProductCard";

export default function Message({ message }) {
  const isUser = message.role === "user";
  const data = message.data;

  return (
    <div className={`message message--${isUser ? "user" : "assistant"}`}>
      <div className="message__avatar">{isUser ? "You" : "AI"}</div>

      <div className="message__body">
        <div className="message__bubble">{message.content}</div>

        {data?.clarification?.options?.length > 0 && (
          <div className="message__options">
            {data.clarification.options.map((option) => (
              <span className="suggestion-chip" key={option}>{option}</span>
            ))}
          </div>
        )}

        {data?.recommendations?.map((product) => (
          <ProductCard key={product.product_id} product={product} />
        ))}

        {data?.upsell && (
          <section className="upsell-section">
            <h3>Optional step-up</h3>
            <ProductCard product={data.upsell} />
          </section>
        )}
      </div>
    </div>
  );
}
