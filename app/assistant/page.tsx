"use client";
import { useEffect, useState, useRef } from "react";
import ReactMarkdown from "react-markdown";
import {
  Send,
  Menu as MenuIcon,
  X,
  Plus,
  History,
  Settings,
  MessageSquare,
  Trash2,
  Edit3,
} from "lucide-react";

export default function Assistant() {
  const [query, setQuery] = useState("");
  const [messages, setMessages] = useState<
    { role: "user" | "assistant"; content: string }[]
  >([]);
  const [isLoading, setIsLoading] = useState(false);
  const [isSidebarOpen, setIsSidebarOpen] = useState(true);
  const [activeTab, setActiveTab] = useState<"history" | "services">("history");
  const [chatHistory, setChatHistory] = useState<
    { id: string; name: string; messages: { role: string; content: string }[] }[]
  >([]);
  const [selectedImage, setSelectedImage] = useState<File | null>(null);
  const [isImageProcessed, setIsImageProcessed] = useState(false);
  const messagesEndRef = useRef<null | HTMLDivElement>(null);

  // Scroll to bottom when messages change
  const scrollToBottom = () => {
    messagesEndRef.current?.scrollIntoView({ behavior: "smooth" });
  };

  useEffect(() => {
    scrollToBottom();
  }, [messages]);

  // Load chat history on mount
  useEffect(() => {
    const savedChats = localStorage.getItem("chatHistory");
    if (savedChats) {
      setChatHistory(JSON.parse(savedChats));
    }
  }, []);

  // Save chat history whenever it changes
  useEffect(() => {
    localStorage.setItem("chatHistory", JSON.stringify(chatHistory));
  }, [chatHistory]);

  // Handle Image Selection
  const handleImageChange = (e: React.ChangeEvent<HTMLInputElement>) => {
    if (e.target.files && e.target.files[0] && !isImageProcessed) {
      setSelectedImage(e.target.files[0]);
    }
  };

  // Handle Image Upload
  const handleImageUpload = async (e: React.FormEvent) => {
    e.preventDefault();
    if (!selectedImage) return;

    const formData = new FormData();
    formData.append("image", selectedImage);

    try {
      const imageResponse = await fetch("http://127.0.0.1:4000/upload_image", {
        method: "POST",
        body: formData,
      });

      if (!imageResponse.ok) {
        throw new Error(`Image upload failed with status: ${imageResponse.status}`);
      }
      // TODO: Get the image and prcess later
      // const imageData = await imageResponse.json();

      setMessages((prev) => [
        ...prev,
        { role: "assistant", content: "Image uploaded and processed successfully." },
      ]);
      setIsImageProcessed(true);
    } catch (error) {
      console.error("Image Upload Error:", error);
      setMessages((prev) => [
        ...prev,
        {
          role: "assistant",
          content: "Sorry, there was an error uploading your image.",
        },
      ]);
    }
  };

  // Handle Chat Submission
  const handleChatSubmit = async (e: React.FormEvent) => {
    e.preventDefault();
    if (!query.trim()) return;

    type Payload = {
      user_answer: string;
    };
    
    const payload: Payload = {
      user_answer: query,
    };
    
    setMessages((prev) => [...prev, { role: "user", content: query }]);
    setQuery("");

    setIsLoading(true);

    try {
      const response = await fetch("http://127.0.0.1:4000/chat", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify(payload),
      });

      if (!response.ok) {
        throw new Error(`HTTP error! status: ${response.status}`);
      }

      const data = await response.json();
      
      // Handle bot question
      if (data.bot_question) {
        setMessages((prev) => [
          ...prev,
          { role: "assistant", content: data.bot_question },
        ]);
      }

      // Handle final response
      if (data.response) {
        setMessages((prev) => [
          ...prev,
          { role: "assistant", content: data.response },
        ]);
      }
    } catch (error) {
      console.error("Error:", error);
      setMessages((prev) => [
        ...prev,
        {
          role: "assistant",
          content: "Sorry, there was an error processing your request.",
        },
      ]);
    } finally {
      setIsLoading(false);
    }
  };

  const startNewChat = () => {
    if (messages.length > 0) {
      const chatName = `Chat ${chatHistory.length + 1}`;
      setChatHistory((prev) => [
        ...prev,
        { id: Date.now().toString(), name: chatName, messages },
      ]);
    }
    setMessages([]);
    setQuery("");
    setSelectedImage(null);
    setIsImageProcessed(false);
  };

  const loadChat = (id: string) => {
    const chat = chatHistory.find((chat) => chat.id === id);
    if (chat) {
      const validMessages = chat.messages.map((message) => ({
        role: message.role as "user" | "assistant",
        content: message.content,
      }));
      setMessages(validMessages);
    }
  };

  const deleteChat = (id: string) => {
    setChatHistory((prev) => prev.filter((chat) => chat.id !== id));
  };

  const renameChat = (id: string) => {
    const newName = prompt("Enter a new name for this chat:");
    if (newName) {
      setChatHistory((prev) =>
        prev.map((chat) =>
          chat.id === id ? { ...chat, name: newName } : chat
        )
      );
    }
  };

  return (
    <div className="flex h-screen bg-black-50">
      <div
        className={`${
          isSidebarOpen ? "w-64" : "w-0"
        } bg-black border-r transition-all duration-300 flex flex-col`}
      >
        <div className="p-4 border-b flex justify-between items-center">
          <h2 className="font-semibold text-lg">Assistant</h2>
          <button
            onClick={() => setIsSidebarOpen(false)}
            className="p-1 hover:bg-black-100 rounded"
          >
            <X className="w-5 h-5" />
          </button>
        </div>

        <button
          onClick={startNewChat}
          className="m-4 p-2 bg-red-500 text-white rounded-lg flex items-center gap-2 hover:bg-red-500"
        >
          <Plus className="w-4 h-4" />
          New Chat
        </button>

        <div className="flex border-b">
          <button
            onClick={() => setActiveTab("history")}
            className={`flex-1 p-3 ${
              activeTab === "history"
                ? " text-red-500"
                : "text-gray-600"
            }`}
          >
            <div className="flex items-center justify-center gap-2">
              <History className="w-4 h-4" />
              History
            </div>
          </button>
          <button
            onClick={() => setActiveTab("services")}
            className={`flex-1 p-3 ${
              activeTab === "services"
                ? " text-red-500"
                : "text-gray-600"
            }`}
          >
            <div className="flex items-center justify-center gap-2">
              <Settings className="w-4 h-4" />
              Services
            </div>
          </button>
        </div>

        <div className="flex-1 overflow-y-auto p-4">
          {activeTab === "history" ? (
            <div className="space-y-2">
              {chatHistory.map((chat) => (
                <div
                  key={chat.id}
                  className="p-2 hover:bg-black-100 rounded cursor-pointer flex items-center justify-between"
                >
                  <div
                    onClick={() => loadChat(chat.id)}
                    className="flex items-center gap-2 flex-grow"
                  >
                    <MessageSquare className="w-4 h-4 text-red-500" />
                    <span className="text-sm">{chat.name}</span>
                  </div>
                  <div className="flex gap-2 ">
                    <button
                      onClick={() => renameChat(chat.id)}
                      className="hover:text-blue-500"
                    >
                      <Edit3 className="w-4 h-4" />
                    </button>
                    <button
                      onClick={() => deleteChat(chat.id)}
                      className="hover:text-red-500"
                    >
                      <Trash2 className="w-4 h-4 " />
                    </button>
                  </div>
                </div>
              ))}
            </div>
          ) : (
            <div className="space-y-2">
              <div className="p-2 hover:bg-black-100 rounded cursor-pointer">
                <h3 className="font-medium">Vehicle Diagnostics</h3>
                <p className="text-sm text-black-600">Check vehicle health</p>
              </div>
              <div className="p-2 hover:bg-black-100 rounded cursor-pointer">
                <h3 className="font-medium">Maintenance Schedule</h3>
                <p className="text-sm text-black-600">View upcoming services</p>
              </div>
            </div>
          )}
        </div>
      </div>

      <div className="flex-1 flex flex-col">
        <div className="bg-black border-b p-4 flex items-center gap-4">
          {!isSidebarOpen && (
            <button
              onClick={() => setIsSidebarOpen(true)}
              className="p-1 hover:bg-black-100 rounded"
            >
              <MenuIcon className="w-5 h-5" />
            </button>
          )}
          <h1 className="font-semibold">Vehicle Assistant</h1>
        </div>

        <div className="flex-1 overflow-y-auto p-4 space-y-4">
          {messages.map((message, index) => (
            <div
              key={index}
              className={`flex ${
                message.role === "user" ? "justify-end" : "justify-start"
              }`}
            >
              <div
                className={`max-w-[80%] rounded-lg p-3 ${
                  message.role === "user"
                    ? "bg-red-500 text-black"
                    : "bg-white shadow-sm border text-black"
                }`}
              >
                {message.role === "assistant" ? (
                  <ReactMarkdown>{message.content}</ReactMarkdown>
                ) : (
                  message.content
                )}
              </div>
            </div>
          ))}
          {isLoading && (
            <div className="flex justify-start items-center space-x-4">
            <div className="animate-spin rounded-full h-6 w-6 border-t-2 border-b-2 border-red"></div>
            <div className="bg-black shadow-sm border rounded-lg p-3 text-white">
              Let&apos;s Go..
            </div>

          </div>
          )}
          <div ref={messagesEndRef} />
        </div>

        <div className="p-4 bg-black border-t flex flex-col gap-4">
          <form onSubmit={handleImageUpload} className="flex gap-2 items-center">
            {!isImageProcessed ? (
              <>
                <input
                  type="file"
                  accept="image/*"
                  onChange={handleImageChange}
                  className="p-2 bg-gray-700 text-white rounded-lg"
                  
                />
                <button
                  type="submit"
                  disabled={!selectedImage}
                  className={`p-2 bg-blue-500 text-white rounded-lg disabled:opacity-50 hover:bg-blue-600 transition-colors ${
                    !selectedImage ? "cursor-not-allowed" : ""
                  }`}
                >
                  Submit Image
                  
                </button>
              </>
            ) : (
              <span className="text-green-500">Image Uploaded</span>
            )}
          </form>

          <form onSubmit={handleChatSubmit} className="flex gap-2 items-center">
            <input
              type="text"
              value={query}
              onChange={(e) => setQuery(e.target.value)}
              placeholder="Ask about your vehicle..."
              className="flex-1 p-2 border rounded-lg focus:outline-none focus:border-black-500 text-black"
            />
            <button
              type="submit"
              disabled={isLoading}
              className="p-2 bg-red-500 text-white rounded-lg disabled:opacity-50 hover:bg-red-600 transition-colors"
            >
              <Send className="w-5 h-5" />
            </button>
          </form>
        </div>
      </div>
    </div>
  );
}