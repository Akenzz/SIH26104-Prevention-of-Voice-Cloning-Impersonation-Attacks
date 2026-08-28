import { IconInfo } from "@/components/ui/icons.jsx";

// Honest framing, straight from the build plan's Claims Ledger. Kept visible so
// the demo never over-promises: this is decision support, not identity proof.
export default function Disclaimer() {
  return (
    <div className="disclaimer">
      <IconInfo size={16} />
      <p>
        <b>Decision support, not proof of identity.</b> This tool estimates the risk that call
        audio is synthetic or manipulated and recommends stepping up verification. It does not
        prove who is calling and does not prevent fraud on its own. Scores are meaningful only on
        audio similar to what the detector was measured on.
      </p>
    </div>
  );
}
