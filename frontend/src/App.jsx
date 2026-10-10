import ChatWindow from "./components/ChatWindow";
import "./App.css";

export default function App() {
  return (
    <main className="app-shell">
      <section className="intro">
        <div className="eyebrow"><span /> SMARTER SHOPPING, SIMPLIFIED</div>
        <h1>Find the right product <span>for you.</span></h1>
        <p>
          Describe what you need. Get recommendations that consider your budget,
          preferences, product details, and potential trade-offs.
        </p>
        <div className="feature-list">
          <span>✓ Budget-aware matching</span>
          <span>✓ Clear product trade-offs</span>
          <span>✓ Catalog-based recommendations</span>
        </div>
      </section>

      <ChatWindow />

      <footer className="app-footer">
        Recommendations are based on the application's catalog. Verify product
        details before purchasing.
      </footer>
    </main>
  );
}
