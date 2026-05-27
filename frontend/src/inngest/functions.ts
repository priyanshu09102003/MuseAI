
import { inngest } from "./client";

export const generateSong = inngest.createFunction(
  { id: "generate-song", triggers: { event: "song-generation" } },
  async ({ event, step }) => {
    const result = await step.run("handle-task", async () => {
      return { processed: true, id: event.data.id };
    });

    await step.sleep("pause", "1s");

    return { message: `Task ${event.data.id} complete`, result };
  }
);