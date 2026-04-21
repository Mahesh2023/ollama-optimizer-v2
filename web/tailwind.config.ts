import type { Config } from "tailwindcss";

const config: Config = {
  content: ["./app/**/*.{ts,tsx}", "./components/**/*.{ts,tsx}"],
  theme: {
    extend: {
      colors: {
        bg: "#0b0e14",
        surface: "#151922",
        border: "#1f2430",
        primary: "#7aa2f7",
        accent: "#bb9af7",
        success: "#9ece6a",
        warning: "#e0af68",
        danger: "#f7768e",
        muted: "#565f89",
      },
      fontFamily: {
        mono: ["JetBrains Mono", "ui-monospace", "monospace"],
      },
    },
  },
  plugins: [],
};

export default config;
