const express = require("express");
const bodyParser = require("body-parser");
const axios = require("axios");

const app = express();
const PORT = 3000;

// Middleware to parse JSON
app.use(bodyParser.json());

// Route for prediction
app.post("/predict", async (req, res) => {
  try {
    const inputData = req.body; // Expecting description, category, status
    console.log("Input received:", inputData);

    // Forward the input data to the Python ML service
    const response = await axios.post(
      "http://localhost:5000/predict",
      inputData
    );

    // Return the prediction from the Python service
    res.json(response.data);
  } catch (error) {
    console.error("Error in Node.js server:", error.message);
    res.status(500).send({ error: error.message });
  }
});

// Start the server
app.listen(PORT, () => {
  console.log(`Node.js server is running at http://localhost:${PORT}`);
});