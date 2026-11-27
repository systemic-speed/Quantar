const path = require("path");
const fs = require("fs");
const vscode = require("vscode");
const crypto = require("crypto");
const { execFile } = require("child_process");
const { promisify } = require("util");
const { LanguageClient } = require("vscode-languageclient/node");

const execFileAsync = promisify(execFile);

let client;
let clientStartPromise;
let extensionContext;
let pylanceActivation = Promise.resolve(false);
let mirrorSessionId;
let pythonRuntime;
let checkDiagnostics;
const checkedDiagnosticUris = new Set();
const pythonMirrors = new Map();
let outputChannel;

function projectVenvPython(projectRoot) {
    const candidates =
        process.platform === "win32"
            ? [path.join(projectRoot, ".venv", "Scripts", "python.exe")]
            : [path.join(projectRoot, ".venv", "bin", "python")];
    return candidates.find((candidate) => fs.existsSync(candidate));
}

/** Work out which Python interpreter should host the language server. */
async function resolvePython(projectRoot, isSourceCheckout) {
    const configured = vscode.workspace
        .getConfiguration("quantar")
        .get("pythonPath");
    if (configured) {
        return { command: configured, args: [] };
    }

    // A source checkout usually has its own virtual environment, which is the
    // only interpreter guaranteed to have Quantar's LSP dependencies.
    if (isSourceCheckout) {
        const venv = projectVenvPython(projectRoot);
        if (venv) {
            return { command: venv, args: [] };
        }
    }

    const pythonExtension = vscode.extensions.getExtension("ms-python.python");
    if (pythonExtension) {
        try {
            const pythonApi = await pythonExtension.activate();
            const workspaceUri = vscode.workspace.workspaceFolders?.[0]?.uri;
            const execution = await pythonApi.settings?.getExecutionDetails?.(
                workspaceUri,
            );
            if (Array.isArray(execution?.execCommand) && execution.execCommand.length > 0) {
                const [command, ...args] = execution.execCommand;
                return { command, args };
            }
            if (typeof execution?.execCommand === "string") {
                return {
                    command: execution.execCommand,
                    args: execution.execArgs || [],
                };
            }
        } catch (error) {
            outputChannel?.appendLine(
                `Could not query the Python extension for an interpreter: ${error.message}`,
            );
        }
    }

    return {
        command: process.platform === "win32" ? "python" : "python3",
        args: [],
    };
}

function quantarPythonEnvironment(projectRoot, isSourceCheckout) {
    const sourceRoot = path.join(projectRoot, "src");
    return {
        ...process.env,
        PYTHONPATH: [
            ...(isSourceCheckout ? [sourceRoot] : []),
            process.env.PYTHONPATH,
        ]
            .filter(Boolean)
            .join(path.delimiter),
    };
}

const quantarSelector = [{ scheme: "file", language: "quantar" }];

async function pythonMirrorFor(document) {
    if (!clientStartPromise || !extensionContext) return undefined;
    try {
        const [, pylanceReady] = await Promise.all([clientStartPromise, pylanceActivation]);
        if (!pylanceReady) return undefined;
        const key = document.uri.toString();
        const originalSource = document.getText();
        let entry = pythonMirrors.get(key);
        if (entry?.originalSource === originalSource) {
            await entry.sync;
            return entry;
        }
        const converted = await client.sendRequest("quantar/convertDocument", {
            source: originalSource,
        });
        const pythonSource = typeof converted === "string"
            ? converted
            : typeof converted?.source === "string"
                ? converted.source
                : originalSource;
        entry = pythonMirrors.get(key) || entry;
        if (!entry) {
            const digest = crypto.createHash("sha256").update(key).digest("hex");
            const workspaceFolder = vscode.workspace.getWorkspaceFolder(document.uri);
            const base = workspaceFolder
                ? workspaceFolder.uri
                : extensionContext.storageUri || extensionContext.globalStorageUri;
            const dir = vscode.Uri.joinPath(
                base,
                ".quantar-pylance-mirrors",
                mirrorSessionId,
            );
            await fs.promises.mkdir(dir.fsPath, { recursive: true });
            entry = {
                uri: vscode.Uri.joinPath(dir, `${digest}.py`),
                sourceUri: document.uri,
                source: undefined,
                sync: Promise.resolve(),
            };
            pythonMirrors.set(key, entry);
        }

        entry.sync = entry.sync.catch(() => {}).then(async () => {
            if (entry.source === pythonSource) return;
            if (entry.source === undefined) {
                await fs.promises.writeFile(entry.uri.fsPath, pythonSource, "utf8");
            } else {
                const mirror = await vscode.workspace.openTextDocument(entry.uri);
                const edit = new vscode.WorkspaceEdit();
                edit.replace(
                    entry.uri,
                    new vscode.Range(new vscode.Position(0, 0), mirror.positionAt(mirror.getText().length)),
                    pythonSource,
                );
                await vscode.workspace.applyEdit(edit);
                const updated = await vscode.workspace.openTextDocument(entry.uri);
                await updated.save();
            }
            entry.source = pythonSource;
            entry.originalSource = originalSource;
        });
        await entry.sync;
        entry.document = await vscode.workspace.openTextDocument(entry.uri);
        if (entry.document.languageId !== "python") {
            entry.document = await vscode.languages.setTextDocumentLanguage(entry.document, "python");
        }
        return entry;
    } catch (error) {
        outputChannel?.appendLine(`Pylance mirror failed: ${error.message}`);
        return undefined;
    }
}

function mapMirrorLocation(item, entry) {
    if (!item) return item;
    if (item.targetUri) {
        return {
            ...item,
            targetUri: item.targetUri.toString() === entry.uri.toString()
                ? entry.sourceUri
                : item.targetUri,
        };
    }
    if (item.uri && item.range) {
        const uri = item.uri.toString() === entry.uri.toString() ? entry.sourceUri : item.uri;
        return new vscode.Location(uri, item.range);
    }
    return item;
}

async function pylanceCommand(document, command, args = []) {
    const entry = await pythonMirrorFor(document);
    if (!entry) return undefined;
    try {
        let result = await vscode.commands.executeCommand(command, entry.uri, ...args);
        if (command === "vscode.executeHoverProvider" && (!result || result.length === 0)) {
            for (let attempt = 0; attempt < 3 && (!result || result.length === 0); attempt++) {
                await new Promise((resolve) => setTimeout(resolve, 150));
                result = await vscode.commands.executeCommand(command, entry.uri, ...args);
            }
            if (!result || result.length === 0) {
                outputChannel?.appendLine(
                    `Pylance returned no hover for ${document.uri.fsPath}:${args[0].line + 1}:${args[0].character + 1}`,
                );
            }
        }
        return result;
    } catch (error) {
        outputChannel?.appendLine(`${command} failed: ${error.message}`);
        return undefined;
    }
}

function registerPythonLanguageFeatures(context) {
    const pylance = vscode.extensions.getExtension("ms-python.vscode-pylance");
    if (pylance) {
        pylanceActivation = pylance.activate().then(() => true).catch((error) => {
            outputChannel.appendLine(`Could not activate Pylance: ${error.message}`);
            return false;
        });
    } else {
        outputChannel.appendLine("Pylance is not installed; Python language features are unavailable.");
    }
    const diagnostics = vscode.languages.createDiagnosticCollection("quantar-pylance");
    context.subscriptions.push(
        diagnostics,
        vscode.languages.onDidChangeDiagnostics((event) => {
            for (const entry of pythonMirrors.values()) {
                if (!event.uris.some((uri) => uri.toString() === entry.uri.toString())) continue;
                const mapped = vscode.languages.getDiagnostics(entry.uri).map((diagnostic) => {
                    const copy = new vscode.Diagnostic(
                        diagnostic.range,
                        diagnostic.message,
                        diagnostic.severity,
                    );
                    copy.code = diagnostic.code;
                    copy.source = diagnostic.source || "Pylance";
                    copy.tags = diagnostic.tags;
                    copy.relatedInformation = diagnostic.relatedInformation?.map((item) =>
                        new vscode.DiagnosticRelatedInformation(
                            mapMirrorLocation(item.location, entry),
                            item.message,
                        ),
                    );
                    return copy;
                });
                diagnostics.set(entry.sourceUri, mapped);
            }
        }),
        vscode.workspace.onDidCloseTextDocument(async (document) => {
            const key = document.uri.toString();
            const entry = pythonMirrors.get(key);
            if (!entry) return;
            pythonMirrors.delete(key);
            diagnostics.delete(entry.sourceUri);
            await fs.promises.unlink(entry.uri.fsPath).catch(() => {});
        }),
        vscode.languages.registerHoverProvider(quantarSelector, {
            async provideHover(document, position) {
                const result = await pylanceCommand(document, "vscode.executeHoverProvider", [position]);
                if (!Array.isArray(result) || result.length === 0) return undefined;
                const contents = result.flatMap((hover) => hover.contents || []);
                return contents.length ? new vscode.Hover(contents) : undefined;
            },
        }),
        vscode.languages.registerDefinitionProvider(quantarSelector, {
            async provideDefinition(document, position) {
                const result = await pylanceCommand(document, "vscode.executeDefinitionProvider", [position]);
                const entry = pythonMirrors.get(document.uri.toString());
                if (!result || !entry) return undefined;
                const items = Array.isArray(result) ? result : [result];
                return items.map((item) => mapMirrorLocation(item, entry));
            },
        }),
        vscode.languages.registerTypeDefinitionProvider(quantarSelector, {
            async provideTypeDefinition(document, position) {
                const result = await pylanceCommand(document, "vscode.executeTypeDefinitionProvider", [position]);
                const entry = pythonMirrors.get(document.uri.toString());
                if (!result || !entry) return undefined;
                const items = Array.isArray(result) ? result : [result];
                return items.map((item) => mapMirrorLocation(item, entry));
            },
        }),
        vscode.languages.registerImplementationProvider(quantarSelector, {
            async provideImplementation(document, position) {
                const result = await pylanceCommand(document, "vscode.executeImplementationProvider", [position]);
                const entry = pythonMirrors.get(document.uri.toString());
                if (!result || !entry) return undefined;
                const items = Array.isArray(result) ? result : [result];
                return items.map((item) => mapMirrorLocation(item, entry));
            },
        }),
        vscode.languages.registerReferenceProvider(quantarSelector, {
            async provideReferences(document, position, referenceContext) {
                const result = await pylanceCommand(document, "vscode.executeReferenceProvider", [position, referenceContext]);
                const entry = pythonMirrors.get(document.uri.toString());
                return Array.isArray(result) && entry
                    ? result.map((item) => mapMirrorLocation(item, entry))
                    : undefined;
            },
        }),
        vscode.languages.registerRenameProvider(quantarSelector, {
            async provideRenameEdits(document, position, newName) {
                const result = await pylanceCommand(document, "vscode.executeDocumentRenameProvider", [position, newName]);
                const entry = pythonMirrors.get(document.uri.toString());
                if (!result || !entry) return undefined;
                const mapped = new vscode.WorkspaceEdit();
                for (const [uri, edits] of result.entries()) {
                    mapped.set(
                        uri.toString() === entry.uri.toString() ? entry.sourceUri : uri,
                        edits,
                    );
                }
                return mapped;
            },
        }),
        vscode.languages.registerDocumentSymbolProvider(quantarSelector, {
            provideDocumentSymbols(document) {
                return pylanceCommand(document, "vscode.executeDocumentSymbolProvider");
            },
        }),
        vscode.languages.registerDocumentHighlightProvider(quantarSelector, {
            provideDocumentHighlights(document, position) {
                return pylanceCommand(document, "vscode.executeDocumentHighlights", [position]);
            },
        }),
        vscode.languages.registerCompletionItemProvider(quantarSelector, {
            async provideCompletionItems(document, position, _token, completionContext) {
                return pylanceCommand(document, "vscode.executeCompletionItemProvider", [
                    position,
                    completionContext.triggerCharacter,
                ]);
            },
        }, ".", "(", ",", "'", '"'),
        vscode.languages.registerSignatureHelpProvider(quantarSelector, {
            async provideSignatureHelp(document, position, _token, signatureContext) {
                return pylanceCommand(document, "vscode.executeSignatureHelpProvider", [
                    position,
                    signatureContext.triggerCharacter,
                    signatureContext.isRetrigger,
                ]);
            },
        }, "(", ","),
    );
}

function parseCheckErrors(stderr) {
    const errors = [];
    for (const rawLine of stderr.split(/\r?\n/)) {
        const line = rawLine.trim();
        if (!line.startsWith("[CHECK] ERROR ")) continue;
        const detail = line.slice("[CHECK] ERROR ".length);
        const separator = detail.lastIndexOf(": ");
        const location = separator === -1 ? detail : detail.slice(0, separator);
        const message = separator === -1 ? "" : detail.slice(separator + 2);
        const match = location.match(/^(.*):(\d+)$/);
        errors.push({
            fsPath: match ? match[1] : location,
            line: match ? Number(match[2]) : 1,
            message,
        });
    }
    return errors;
}

function clearCheckDiagnostics() {
    if (!checkDiagnostics) return;
    for (const uri of checkedDiagnosticUris) {
        checkDiagnostics.delete(uri);
    }
    checkedDiagnosticUris.clear();
}

async function publishCheckErrors(errors) {
    clearCheckDiagnostics();
    if (!checkDiagnostics) return;
    for (const error of errors) {
        const uri = vscode.Uri.file(path.resolve(error.fsPath));
        const line = Math.max(0, error.line - 1);
        const range = new vscode.Range(new vscode.Position(line, 0), new vscode.Position(line, 0));
        const diagnostic = new vscode.Diagnostic(
            range,
            error.message || "Quantar check error",
            vscode.DiagnosticSeverity.Error,
        );
        diagnostic.source = "qtr check";
        checkDiagnostics.set(uri, [...(checkDiagnostics.get(uri) || []), diagnostic]);
        checkedDiagnosticUris.add(uri);
    }
}

async function checkActiveDocument() {
    const editor = vscode.window.activeTextEditor;
    if (!editor || editor.document.languageId !== "quantar") {
        void vscode.window.showWarningMessage("Open a Quantar file to run qtr check.");
        return;
    }
    if (editor.document.isUntitled || editor.document.isDirty) {
        void vscode.window.showWarningMessage("Save the Quantar file before running qtr check.");
        return;
    }
    if (!pythonRuntime) {
        void vscode.window.showErrorMessage("Quantar is not ready yet. Try again in a moment.");
        return;
    }

    const file = editor.document.uri.fsPath;
    const workspaceFolder = vscode.workspace.getWorkspaceFolder(editor.document.uri);
    const cwd = workspaceFolder ? workspaceFolder.uri.fsPath : path.dirname(file);
    const args = [...pythonRuntime.args, "-m", "lang_systems.runner", "check", file];
    const options = {
        cwd,
        env: quantarPythonEnvironment(pythonRuntime.projectRoot, pythonRuntime.isSourceCheckout),
        windowsHide: true,
        maxBuffer: 10 * 1024 * 1024,
    };

    try {
        const { stdout, stderr } = await execFileAsync(pythonRuntime.command, args, options);
        clearCheckDiagnostics();
        const summary = (stdout || stderr || "").trim();
        void vscode.window.showInformationMessage(summary || "Quantar check passed.");
    } catch (error) {
        const stderr = error.stderr ? String(error.stderr) : "";
        const stdout = error.stdout ? String(error.stdout) : "";
        const errors = parseCheckErrors(stderr);
        await publishCheckErrors(errors);
        if (errors.length === 0) {
            const detail = (stderr || stdout || error.message || "").trim();
            outputChannel?.appendLine(`qtr check failed: ${detail}`);
            void vscode.window.showErrorMessage("Quantar check failed. See the Quantar output channel for details.");
        }
    }
}

/** @param {vscode.ExtensionContext} context */
async function activate(context) {
    extensionContext = context;
    mirrorSessionId = crypto.randomBytes(6).toString("hex");
    const projectRoot = path.join(context.extensionPath, "..");
    const sourceRoot = path.join(projectRoot, "src");
    const isSourceCheckout = fs.existsSync(
        path.join(sourceRoot, "lang_systems", "lsp"),
    );

    const runtime = await resolvePython(projectRoot, isSourceCheckout);
    pythonRuntime = { ...runtime, projectRoot, isSourceCheckout };

    const serverOptions = {
        command: runtime.command,
        args: [...runtime.args, "-u", "-m", "lang_systems.lsp"],
        options: {
            cwd: isSourceCheckout
                ? projectRoot
                : vscode.workspace.workspaceFolders?.[0]?.uri.fsPath ||
                  context.extensionPath,
            env: quantarPythonEnvironment(projectRoot, isSourceCheckout),
        },
    };

    checkDiagnostics = vscode.languages.createDiagnosticCollection("quantar-check");
    outputChannel = vscode.window.createOutputChannel("Quantar");
    client = new LanguageClient(
        "quantarLanguageServer",
        "Quantar Language Server",
        serverOptions,
        {
            documentSelector: [{ scheme: "file", language: "quantar" }],
            outputChannel,
        },
    );
    context.subscriptions.push(
        client,
        outputChannel,
        checkDiagnostics,
        vscode.commands.registerCommand("quantar.check", checkActiveDocument),
    );
    registerPythonLanguageFeatures(context);
    clientStartPromise = client.start();
    clientStartPromise.catch((error) => {
        outputChannel.appendLine(
            `Failed to start the language server: ${error.message}`,
        );
        outputChannel.show(true);
        const installCommand = isSourceCheckout
            ? 'uv sync --extra lsp (from the project root)'
            : 'python -m pip install "quantar[lsp]"';
        void vscode.window.showErrorMessage(
            `Quantar could not start its language server. Install Quantar with LSP support in the selected Python environment: ${installCommand}. Then reload VS Code.`,
        );
    });
}

async function deactivate() {
    clearCheckDiagnostics();
    await client?.stop();
    const directories = new Set();
    await Promise.all(
        [...pythonMirrors.values()].map((entry) => {
            directories.add(path.dirname(entry.uri.fsPath));
            return fs.promises.unlink(entry.uri.fsPath).catch(() => {});
        }),
    );
    pythonMirrors.clear();
    await Promise.all(
        [...directories].map((directory) =>
            fs.promises.rmdir(directory).catch(() => {}),
        ),
    );
}

module.exports = { activate, deactivate };
