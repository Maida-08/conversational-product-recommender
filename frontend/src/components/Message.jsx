
import ReactMarkdown from "react-markdown";
import ProductCard from "./ProductCard";

export default function Message({ message }) {
  const isUser = message.role === "user";
  const data = message.data;

    // Remove internal match markers from shopper-facing text.
  const content = (message.content || "")
    .replace(
      /\*{0,2}(?:primary|secondary|caution|upsell)(?:strong|partial|conflict|poor)\*{0,2}/gi,
    )
    .replace(/\n{3,}/g, "\n\n")
    .trim();

  return (
    <div className={`message message--${isUser ? "user" : "assistant"}`}>
      <div className="message__avatar">{isUser ? "You" : "AI"}</div>

      <div className="message__body">
        <div className="message__bubble">
          {isUser ? (
            content
          ) : (
            <ReactMarkdown>{content}</ReactMarkdown>
          )}
        </div>

        {data?.clarification?.options?.length > 0 && (
          <div className="message__options">
            {data.clarification.options.map((option) => (
              <span className="suggestion-chip" key={option}>
                {option}
              </span>
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
