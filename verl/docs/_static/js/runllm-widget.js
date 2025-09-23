/**
 * RunLLM Widget Integration Script
 * 
 * This script dynamically loads the RunLLM chatbot widget for the VERL documentation site.
 * It creates and configures a script element to embed the interactive AI assistant.
 */

// Wait for the DOM to be fully loaded before executing

document.addEventListener("DOMContentLoaded", function () {
  var script = document.createElement("script");
  script.type = "module";
  script.id = "runllm-widget-script";
  script.src = "https://widget.runllm.com";
  script.setAttribute("version", "stable");
  script.setAttribute("crossorigin", "true");
  script.setAttribute("runllm-keyboard-shortcut", "Mod+j");
  script.setAttribute("runllm-name", "verl Chatbot");
  script.setAttribute("runllm-position", "TOP_RIGHT");
  script.setAttribute("runllm-assistant-id", "679");
  script.async = true;
  document.head.appendChild(script);
});
