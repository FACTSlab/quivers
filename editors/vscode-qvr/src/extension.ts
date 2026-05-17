/**
 * VS Code extension entry point for QVR.
 *
 * Starts a `vscode-languageclient` against the `qvr-lsp` executable
 * shipped by the `quivers[lsp]` Python extra. The TM grammar continues
 * to handle initial highlighting; semantic tokens and diagnostics
 * arrive from the language server.
 */

import * as vscode from "vscode";
import {
  LanguageClient,
  LanguageClientOptions,
  ServerOptions,
  TransportKind,
} from "vscode-languageclient/node";

let client: LanguageClient | undefined;

export function activate(context: vscode.ExtensionContext): void {
  const config = vscode.workspace.getConfiguration("qvr");
  if (!config.get<boolean>("lsp.enabled", true)) {
    return;
  }
  const command = config.get<string>("lsp.path", "qvr-lsp");
  const args = config.get<string[]>("lsp.args", []);

  const serverOptions: ServerOptions = {
    command,
    args,
    transport: TransportKind.stdio,
  };
  const clientOptions: LanguageClientOptions = {
    documentSelector: [{ scheme: "file", language: "qvr" }],
    synchronize: {
      fileEvents: vscode.workspace.createFileSystemWatcher("**/*.qvr"),
    },
  };

  client = new LanguageClient(
    "qvr",
    "QVR Language Server",
    serverOptions,
    clientOptions
  );
  context.subscriptions.push({
    dispose: () => {
      void client?.stop();
    },
  });
  client.start();
}

export function deactivate(): Thenable<void> | undefined {
  return client?.stop();
}
