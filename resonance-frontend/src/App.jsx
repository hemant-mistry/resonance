import React, { useState, useEffect, useRef } from 'react';

const App = () => {
  const [text, setText] = useState('');
  const [imageSrc, setImageSrc] = useState(null);
  const [emotionState, setEmotionState] = useState('IDLE');
  const [isConnected, setIsConnected] = useState(false);
  const [isGenerating, setIsGenerating] = useState(false); // NEW: Loading state
  
  const wsRef = useRef(null);
  const typingTimeoutRef = useRef(null);

  useEffect(() => {
    const ws = new WebSocket('ws://localhost:8000/ws');
    ws.onopen = () => setIsConnected(true);
    
    ws.onmessage = (event) => {
      const data = JSON.parse(event.data);
      if (data.image_data) {
        setImageSrc(data.image_data);
        setIsGenerating(false); // NEW: Turn off loading when image arrives
      }
      if (data.emotion) setEmotionState(`${data.emotion} (${data.confidence})`);
    };
    
    ws.onclose = () => setIsConnected(false);
    wsRef.current = ws;
    return () => ws.close();
  }, []);

  const handleTextChange = (e) => {
    const newText = e.target.value;
    setText(newText);
    
    // Clear the timer if they are still typing
    if (typingTimeoutRef.current) clearTimeout(typingTimeoutRef.current);
    
    // NEW DEBOUNCE: Wait 2.5 seconds after they STOP typing before sending
    typingTimeoutRef.current = setTimeout(() => {
      if (wsRef.current?.readyState === WebSocket.OPEN && newText.trim().length > 2) {
        setIsGenerating(true); // Turn on the loading UI
        wsRef.current.send(newText);
      }
    }, 2500); 
  };

  const handleDownload = () => {
    if (!imageSrc) return;
    const link = document.createElement('a');
    link.href = imageSrc;
    link.download = `mindscape_${emotionState.split(' ')[0]}_${Date.now()}.webp`;
    document.body.appendChild(link);
    link.click();
    document.body.removeChild(link);
  };

  return (
    <div className="relative w-screen h-screen bg-neutral-950 overflow-hidden font-sans text-white flex">
      
      {/* Background Blur */}
      {imageSrc && (
        <img 
          src={imageSrc} 
          alt="background"
          className={`absolute inset-0 w-full h-full object-cover opacity-30 transition-all duration-1000 ${isGenerating ? 'blur-3xl brightness-50' : 'blur-[40px]'}`}
        />
      )}

      {/* Main Content Split Layout */}
      <div className="relative z-10 w-full h-full flex flex-col md:flex-row p-8 gap-8 items-center justify-center max-w-7xl mx-auto">
        
        {/* Left Column: The Journal Input */}
        <div className="w-full md:w-1/2 flex flex-col h-[60vh]">
          <div className="flex-grow bg-black/60 backdrop-blur-xl border border-white/10 rounded-2xl p-8 shadow-2xl flex flex-col">
            <textarea 
              className="w-full h-full bg-transparent resize-none outline-none text-3xl font-light leading-relaxed placeholder:text-white/20 text-white"
              placeholder="Write your thoughts. Pause for a moment to let the canvas paint..."
              value={text}
              onChange={handleTextChange}
              autoFocus
            />
          </div>
          <div className="mt-4 flex justify-between items-center text-xs font-mono tracking-widest text-white/50 px-2">
            <div className="flex items-center gap-2">
              <span className={`w-2 h-2 rounded-full ${isConnected ? 'bg-green-500 shadow-[0_0_10px_#22c55e]' : 'bg-red-500'}`}></span>
              {isConnected ? 'LINK ACTIVE' : 'DISCONNECTED'}
            </div>
            <span>STATE: {emotionState}</span>
          </div>
        </div>

        {/* Right Column: The Canvas display */}
        <div className="w-full md:w-1/2 flex flex-col items-center justify-center h-[60vh]">
          {imageSrc ? (
            <div className="relative group rounded-2xl overflow-hidden shadow-2xl border border-white/10 w-full aspect-square max-w-[512px]">
              
              <img 
                src={imageSrc} 
                alt="Generated Emotion" 
                className={`w-full h-full object-cover transition-all duration-700 ${isGenerating ? 'scale-105 blur-sm opacity-50' : 'scale-100 blur-0 opacity-100'}`}
              />
              
              {/* Generating Overlay */}
              {isGenerating && (
                <div className="absolute inset-0 flex flex-col items-center justify-center bg-black/40 backdrop-blur-sm z-20">
                  <div className="animate-spin w-8 h-8 border-4 border-white border-t-transparent rounded-full mb-4"></div>
                  <p className="font-mono text-sm tracking-widest animate-pulse">PAINTING THOUGHTS...</p>
                </div>
              )}
              
              {/* Hover Overlay with Download Button (Only shows when NOT generating) */}
              {!isGenerating && (
                <div className="absolute inset-0 bg-black/50 opacity-0 group-hover:opacity-100 transition-opacity flex items-center justify-center backdrop-blur-sm z-30">
                  <button 
                    onClick={handleDownload}
                    className="px-6 py-3 bg-white text-black font-semibold rounded-full hover:bg-neutral-200 transition-colors shadow-lg"
                  >
                    Save Canvas
                  </button>
                </div>
              )}
            </div>
          ) : (
            <div className="w-full aspect-square max-w-[512px] rounded-2xl border border-dashed border-white/20 flex flex-col items-center justify-center text-white/30 space-y-4">
               {isGenerating ? (
                 <>
                   <div className="animate-spin w-8 h-8 border-4 border-white/30 border-t-white rounded-full"></div>
                   <p className="font-mono text-sm tracking-widest animate-pulse">IGNITING NEURAL ENGINE...</p>
                 </>
               ) : (
                 <>
                   <div className="w-12 h-12 rounded-full bg-white/10 animate-pulse"></div>
                   <p className="font-mono text-sm tracking-widest">AWAITING THOUGHTS</p>
                 </>
               )}
            </div>
          )}
        </div>

      </div>
    </div>
  );
};

export default App;