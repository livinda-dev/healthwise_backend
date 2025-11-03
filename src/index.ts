import express from "express";
import dotenv from "dotenv";
import { runFlow } from "@genkit-ai/flow";
import { healthAssistant } from "./flows/healthAssistant.js";

dotenv.config();

const app = express();
app.use(express.json());

app.post("/healthAssistant", async (req, res) => {
  try {
    console.log("🟢 Incoming request body:", JSON.stringify(req.body, null, 2)); // 👈 ADD THIS
    const input = req.body.input || req.body;
    console.log("🟡 Using input for flow:", JSON.stringify(input, null, 2)); // 👈 ADD THIS

    const result = await runFlow(healthAssistant, input);
    res.json(result);
  } catch (err) {
    console.error("❌ Error running flow:", err);
    res
      .status(500)
      .json({ error: (err as Error).message || "Internal Server Error" });
  }
});


const port = process.env.PORT || 8080;
app.listen(port, () => {
  console.log(`🚀 Genkit flow running at http://localhost:${port}/healthAssistant`);
});
