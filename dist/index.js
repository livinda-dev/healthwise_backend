import express from "express";
import dotenv from "dotenv";
import 'dotenv/config';
import { runFlow } from "@genkit-ai/flow";
import { healthAssistant } from "./flows/healthAssistant.js";
dotenv.config();
const app = express();
app.use(express.json());
app.post("/healthAssistant", async (req, res) => {
    try {
        console.log("🟢 Incoming request body:", JSON.stringify(req.body, null, 2));
        // ✅ Handles all cases: Flutter (data.messages), Genkit, or plain
        const input = req.body.data?.messages
            ? req.body.data
            : req.body.input?.input ||
                req.body.input ||
                req.body;
        console.log("🟡 Using input for flow:", JSON.stringify(input, null, 2));
        const result = await runFlow(healthAssistant, input);
        res.json(result);
    }
    catch (err) {
        console.error("❌ Error running flow:", err);
        res
            .status(500)
            .json({ error: err.message || "Internal Server Error" });
    }
});
const port = process.env.PORT || 8080;
app.listen(port, () => {
    console.log(`🚀 Genkit flow running at http://localhost:${port}/healthAssistant`);
});
