import fs from "fs";
import "dotenv/config";
import { CharacterTextSplitter } from "@langchain/textsplitters";
import { createClient } from "@supabase/supabase-js";
import { SupabaseVectorStore } from "@langchain/community/vectorstores/supabase";

class FakeEmbeddings {
  // For multiple documents
  async embedDocuments(documents) {
    // return a fixed vector (length 1536 is standard for OpenAI embeddings)
    return documents.map(() => Array(1536).fill(0));
  }

  // For a single query
  async embedQuery(query) {
    return Array(1536).fill(0);
  }
}
async function main() {
  try {
    //Read local file
    const text = fs.readFileSync("scrimba-info.txt", "utf-8");

    //Split text into chunks
    const splitter = new CharacterTextSplitter({
      chunkSize: 300,
      chunkOverlap: 50,
      separators: ["\n\n", "\n", " ", ""],
    });

    const documents = await splitter.createDocuments([text]);

    //Initialize Supabase client
    const sbUrl = process.env.SUPABASE_URL;
    const sbApiKey = process.env.SUPABASE_KEY;
    const client = createClient(sbUrl, sbApiKey);
    const openAIApiKey = process.env.OPEN_API_KEY;

    const embeddings = new FakeEmbeddings();

    await SupabaseVectorStore.fromDocuments(documents, embeddings, {
      client,
      tableName: "documents",
    });

    console.log("Successfully uploaded documents to Supabase Vector Store!");
  } catch (err) {
    if (err.name === "InsufficientQuotaError") {
      console.error("OpenAI quota exceeded. Check your plan and usage.");
    } else {
      console.error("Error:", err);
    }
  }
}

main();
