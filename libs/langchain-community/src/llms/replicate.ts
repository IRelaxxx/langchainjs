import { CallbackManagerForLLMRun } from "@langchain/core/callbacks/manager";
import { LLM, type BaseLLMParams } from "@langchain/core/language_models/llms";
import { GenerationChunk } from "@langchain/core/outputs";
import { getEnvironmentVariable } from "@langchain/core/utils/env";
import { Prediction } from "replicate";
import { convertEventStreamToIterableReadableDataStream } from "../utils/event_source_parse.js";

/**
 * Interface defining the structure of the input data for the Replicate
 * class. It includes details about the model to be used, any additional
 * input parameters, and the API key for the Replicate service.
 */
export interface ReplicateInput {
  // owner/model_name:version
  model: `${string}/${string}:${string}`;

  input?: {
    // different models accept different inputs
    [key: string]: string | number | boolean;
  };

  apiKey?: string;

  /** The key used to pass prompts to the model. */
  promptKey?: string;

  streaming?: boolean;
}

/**
 * Class responsible for managing the interaction with the Replicate API.
 * It handles the API key and model details, makes the actual API calls,
 * and converts the API response into a format usable by the rest of the
 * LangChain framework.
 * @example
 * ```typescript
 * const model = new Replicate({
 *   model: "replicate/flan-t5-xl:3ae0799123a1fe11f8c89fd99632f843fc5f7a761630160521c4253149754523",
 * });
 *
 * const res = await model.call(
 *   "Question: What would be a good company name for a company that makes colorful socks?\nAnswer:"
 * );
 * console.log({ res });
 * ```
 */
export class Replicate extends LLM implements ReplicateInput {
  static lc_name() {
    return "Replicate";
  }

  get lc_secrets(): { [key: string]: string } | undefined {
    return {
      apiKey: "REPLICATE_API_TOKEN",
    };
  }

  lc_serializable = true;

  model: ReplicateInput["model"];

  input: ReplicateInput["input"];

  apiKey: string;

  promptKey?: string;

  streaming = false;

  constructor(fields: ReplicateInput & BaseLLMParams) {
    super(fields);

    const apiKey =
      fields?.apiKey ??
      getEnvironmentVariable("REPLICATE_API_KEY") ?? // previous environment variable for backwards compatibility
      getEnvironmentVariable("REPLICATE_API_TOKEN"); // current environment variable, matching the Python library

    if (!apiKey) {
      throw new Error(
        "Please set the REPLICATE_API_TOKEN environment variable"
      );
    }

    this.apiKey = apiKey;
    this.model = fields.model;
    this.input = fields.input ?? {};
    this.promptKey = fields.promptKey;
    this.streaming = fields?.streaming ?? this.streaming;
  }

  _llmType() {
    return "replicate";
  }

  /** @ignore */
  async _request(
    prompt: string,
    options: this["ParsedCallOptions"],
    stream?: boolean
  ): Promise<object | Prediction> {
    const imports = await Replicate.imports();

    const replicate = new imports.Replicate({
      userAgent: "langchain",
      auth: this.apiKey,
    });

    if (this.promptKey === undefined) {
      const [modelString, versionString] = this.model.split(":");
      const version = await replicate.models.versions.get(
        modelString.split("/")[0],
        modelString.split("/")[1],
        versionString
      );
      const openapiSchema = version.openapi_schema;
      const inputProperties: { "x-order": number | undefined }[] =
        // eslint-disable-next-line @typescript-eslint/no-explicit-any
        (openapiSchema as any)?.components?.schemas?.Input?.properties;
      if (inputProperties === undefined) {
        this.promptKey = "prompt";
      } else {
        const sortedInputProperties = Object.entries(inputProperties).sort(
          ([_keyA, valueA], [_keyB, valueB]) => {
            const orderA = valueA["x-order"] || 0;
            const orderB = valueB["x-order"] || 0;
            return orderA - orderB;
          }
        );
        this.promptKey = sortedInputProperties[0][0] ?? "prompt";
      }
    }
    const output = await this.caller.callWithOptions(
      { signal: options.signal },
      () =>
        stream
          ? replicate.predictions.create({
              version: this.model.split(":")[1],
              stream: true,
              input: {
                // eslint-disable-next-line @typescript-eslint/no-non-null-assertion
                [this.promptKey!]: prompt,
                ...this.input,
              },
            })
          : replicate.run(this.model, {
              input: {
                // eslint-disable-next-line @typescript-eslint/no-non-null-assertion
                [this.promptKey!]: prompt,
                ...this.input,
              },
            })
    );
    return output;
  }

  async *_streamResponseChunks(
    prompt: string,
    options: this["ParsedCallOptions"],
    runManager?: CallbackManagerForLLMRun
  ): AsyncGenerator<GenerationChunk> {
    const response: Prediction = (await this._request(
      prompt,
      options,
      true
    )) as Prediction;
    const url = response.urls?.stream;

    if (!url) {
      if (response.error) throw new Error(response.error);
      else throw new Error("Missing stream URL in Replicate response");
    }

    const eventStream = await fetch(url, {
      method: "GET",
      headers: {
        Accept: "text/event-stream",
      },
    });

    let readableStream;
    if (eventStream.ok) {
      if (eventStream.body) {
        readableStream = eventStream.body;
      } else {
        readableStream = new ReadableStream({
          start(controller) {
            controller.error(new Error("Response error: No response body"));
          },
        });
      }
    } else {
      readableStream = new ReadableStream({
        start(controller) {
          controller.error(new Error("Response error: response not ok"));
        },
      });
    }

    const stream =
      convertEventStreamToIterableReadableDataStream(readableStream);
    for await (const chunk of stream) {
      const generationChunk = new GenerationChunk({
        text: chunk,
      });
      yield generationChunk;
      // eslint-disable-next-line no-void
      void runManager?.handleLLMNewToken(generationChunk.text ?? "");
    }
  }

  async _call(
    prompt: string,
    options: this["ParsedCallOptions"],
    runManager?: CallbackManagerForLLMRun
  ): Promise<string> {
    if (!this.streaming) {
      const response = await this._request(prompt, options, false);

      if (typeof response === "string") {
        return response;
      } else if (Array.isArray(response)) {
        return response.join("");
      } else {
        // Note this is a little odd, but the output format is not consistent
        // across models, so it makes some amount of sense.
        return String(response);
      }
    } else {
      const stream = this._streamResponseChunks(prompt, options, runManager);
      let finalResult: GenerationChunk | undefined;
      for await (const chunk of stream) {
        if (finalResult === undefined) {
          finalResult = chunk;
        } else {
          finalResult = finalResult.concat(chunk);
        }
      }
      return finalResult?.text ?? "";
    }
  }

  /** @ignore */
  static async imports(): Promise<{
    Replicate: typeof import("replicate").default;
  }> {
    try {
      const { default: Replicate } = await import("replicate");
      return { Replicate };
    } catch (e) {
      throw new Error(
        "Please install replicate as a dependency with, e.g. `yarn add replicate`"
      );
    }
  }
}
