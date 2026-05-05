"use client";

import { useState } from "react";
import "bootstrap-icons/font/bootstrap-icons.css";

export default function Home() {
  const [open, setOpen] = useState(false);//sorting the chat box conversation
  const [message, setMessage] = useState(""); //storing the current message being typed
  const [chat, setChat] = useState<any[]>([]);//storing the entire chat history as an array of messages
  const [loading, setLoading] = useState(false);//is the chat opened or closed

  const sendMessage = async () => {
    if (!message.trim()) return;

    const newChat = [...chat, { role: "user", content: message }];
    setChat(newChat);
    setMessage("");
    setLoading(true);

    const res = await fetch("/api/chat", {
      method: "POST",
      headers: {
        "Content-Type": "application/json",
      },
      body: JSON.stringify({ messages: newChat }),
    });

    const data = await res.json();

    setChat([
      ...newChat,
      { role: "assistant", content: data.reply },
    ]);

    setLoading(false);
  };

  return (
    <div className="min-h-screen flex items-center justify-center bg-gradient-to-br from-blue-900 via-indigo-800 to-black p-4">

      {/* OPEN BUTTON */}
      {!open && (
        <button
          onClick={() => setOpen(true)}
          className="px-8 py-4 rounded-full bg-white/10 backdrop-blur-md text-white border border-white/20 shadow-xl hover:scale-105 transition flex items-center gap-2"
        >
          <i className="bi bi-car-front-fill text-xl"></i>
          Ask Driving Assistant
        </button>
      )}

      {/* CHAT BOX */}
      {open && (
        <div className="w-full max-w-md h-[600px] bg-white/10 backdrop-blur-xl border border-white/20 rounded-3xl shadow-2xl flex flex-col overflow-hidden">

          {/* HEADER */}
          <div className="p-4 text-white text-center font-bold text-lg border-b border-white/10 flex items-center justify-center gap-2">
            <i className="bi bi-car-front-fill text-2xl"></i>
            Driving AI Assistant
          </div>

          {/* CHAT */}
          <div className="flex-1 p-4 overflow-y-auto flex flex-col gap-3">

            {chat.map((m, i) => (
              <div
                key={i}
                className={`max-w-[80%] px-4 py-2 rounded-2xl text-sm shadow-md ${
                  m.role === "user"
                    ? "bg-blue-500 text-white self-end"
                    : "bg-white text-black self-start"
                }`}
              >
                {m.content}
              </div>
            ))}

            {loading && (
              <div className="self-start bg-white text-black px-4 py-2 rounded-2xl text-sm animate-pulse">
                typing...
              </div>
            )}
          </div>

          {/* INPUT */}
          <div className="p-3 flex gap-2 border-t border-white/10">

            <input
              value={message}
              onChange={(e) => setMessage(e.target.value)}
              placeholder="Ask about driving rules..."
              className="flex-1 px-4 py-2 rounded-xl bg-white text-black outline-none text-sm"
            />

            <button
              onClick={sendMessage}
              className="px-4 py-2 rounded-xl bg-blue-500 text-white hover:bg-blue-600 transition"
            >
              Send
            </button>

          </div>

        </div>
      )}

    </div>
  );
}