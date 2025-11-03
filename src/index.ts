import express from "express";
import dotenv from "dotenv";
import { runFlow } from "@genkit-ai/flow";
import { healthAssistant } from "./flows/healthAssistant.js";

dotenv.config();

const app = express();
app.use(express.json());

// POST endpoint that manually executes the flow
app.post("/healthAssistant", async (req, res) => {
  try {
    const input = req.body.input || req.body; // <-- add this line
    const result = await runFlow(healthAssistant, input);
    res.json(result);
  } catch (err) {
    console.error("Error running flow:", err);
    res.status(500).json({ error: (err as Error).message || "Internal Server Error" });
  }
});


const port = process.env.PORT || 8080;
app.listen(port, () => {
  console.log(`🚀 Genkit flow running at http://localhost:${port}/healthAssistant`);
});
