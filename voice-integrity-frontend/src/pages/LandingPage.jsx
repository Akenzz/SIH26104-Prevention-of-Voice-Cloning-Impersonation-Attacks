import React, { useEffect, useState } from 'react';
import { Link } from 'react-router-dom';
import { Download, ArrowRight, Zap, Search, Activity, ExternalLink, ShieldAlert } from 'lucide-react';

export default function LandingPage() {
  const [scrolled, setScrolled] = useState(false);

  useEffect(() => {
    const handleScroll = () => {
      setScrolled(window.scrollY > 20);
    };
    window.addEventListener('scroll', handleScroll);
    return () => window.removeEventListener('scroll', handleScroll);
  }, []);

  return (
    <div className="min-h-screen bg-[#08090B] text-zinc-100 font-sans selection:bg-blue-500/30 overflow-x-hidden relative">

      {/* Navbar */}
      <nav
        className={`fixed top-6 left-1/2 -translate-x-1/2 w-[calc(100%-2rem)] max-w-[1200px] z-50 transition-all duration-300 rounded-2xl border border-white/10 ${scrolled
          ? 'bg-[#14161C]/80 backdrop-blur-[50px] shadow-[0_8px_30px_rgba(0,0,0,0.5)]'
          : 'bg-[#14161C]/55 backdrop-blur-[50px]'
          }`}
        style={{
          boxShadow: 'inset 0px 1px 0px rgba(255,255,255,0.08)'
        }}
      >
        <div className="px-6 py-4 flex items-center justify-between">
          {/* Logo */}
          <div className="flex items-center gap-3">
            <img src="/logo.png" alt="Voice Integrity" className="h-8 object-contain" />
            <span className="font-heading font-bold text-xl tracking-tight text-white">Voice Integrity</span>
          </div>

          {/* Desktop Links */}
          <div className="hidden md:flex items-center gap-8 text-sm font-medium text-zinc-300">
            <a href="#features" className="hover:text-white transition-colors">Features</a>
            <a href="#how-it-works" className="hover:text-white transition-colors">How It Works</a>
            <a href="#about" className="hover:text-white transition-colors">About</a>
          </div>

          {/* CTA */}
          <div className="flex items-center gap-4">
            <Link
              to="/platform"
              className="hidden sm:flex items-center gap-2 text-sm font-medium text-zinc-300 hover:text-white transition-colors group"
            >
              Try Platform <ArrowRight size={16} className="group-hover:translate-x-0.5 transition-transform" />
            </Link>
          </div>
        </div>
      </nav>

      {/* Main Content */}
      <main className="relative z-10">

        {/* Hero Section */}
        <section className="relative min-h-[95vh] flex items-center pt-32 pb-20 overflow-hidden">
          {/* Hero Background Atmospheric Glows */}
          <div className="absolute inset-0 pointer-events-none z-0">
            {/* Soft blue glow behind the orb on the right */}
            <div className="absolute top-[20%] right-[-10%] w-[800px] h-[800px] rounded-full bg-[#0084FF]/15 blur-[120px]" />
            {/* Very subtle cyan glow in the center */}
            <div className="absolute top-[40%] left-[30%] w-[500px] h-[500px] rounded-full bg-[#319AFF]/5 blur-[150px]" />
            {/* Subtle navy background base */}
            <div className="absolute inset-0 bg-gradient-to-br from-transparent via-[#08090B]/80 to-[#030614]/90" />
          </div>

          <div className="max-w-[1400px] w-full mx-auto px-6 relative z-10">
            <div className="flex flex-col lg:flex-row items-center justify-between gap-12 lg:gap-8">

              {/* Hero Copy (Left 50%) */}
              <div className="w-full lg:w-[45%] flex flex-col items-start animate-fade-in-up mt-8 lg:mt-0">
                <h1 className="font-heading font-bold text-[56px] sm:text-[72px] lg:text-[80px] leading-[1.05] tracking-tight mb-6 text-white" style={{ letterSpacing: '-0.03em' }}>
                  Detect the Voice.<br />Expose the Fake.
                </h1>

                <p className="text-[18px] text-zinc-300 mb-10 leading-relaxed font-sans max-w-[560px]">
                  Voice Integrity uses advanced AI to detect synthetic, cloned, and manipulated speech helping you identify fraudulent voices before they become a threat.
                </p>

                <div className="flex flex-col sm:flex-row items-center gap-5 w-full sm:w-auto">
                  <a
                    href="https://codequantum.in/downloads/app-debug.apk"
                    download
                    className="w-full sm:w-auto flex items-center justify-center gap-2 px-8 py-4 rounded-[16px] bg-[#0084FF] text-white font-medium transition-all duration-300 hover:scale-[1.02] hover:bg-[#319AFF] shadow-[0_0_30px_rgba(0,132,255,0.25)] relative overflow-hidden group"
                  >
                    <div className="absolute inset-0 bg-gradient-to-b from-white/20 to-transparent opacity-50 pointer-events-none" />
                    <Download size={20} className="relative z-10" />
                    <span className="relative z-10 text-[16px]">Download App</span>
                  </a>
                  <Link
                    to="/platform"
                    className="w-full sm:w-auto flex items-center justify-center gap-2 px-8 py-4 rounded-[16px] bg-transparent border border-white/20 hover:border-white/40 hover:bg-white/5 text-white font-medium transition-all duration-300 hover:scale-[1.02] group"
                  >
                    <span className="text-[16px]">Try the Platform</span>
                    <ArrowRight size={20} className="group-hover:translate-x-1 transition-transform" />
                  </Link>
                </div>
              </div>

              {/* Hero Visual (Right 50%) */}
              <div className="w-full lg:w-[55%] flex justify-center lg:justify-end relative">
                <div className="relative w-full max-w-[700px] lg:max-w-[850px] lg:mr-[-100px] aspect-square flex items-center justify-center animate-fade-in-slow pointer-events-none">
                  {/* The Liquid Glass Orb (Video Asset with CSS Filters for Blue Tint) */}
                  <video
                    src="https://future.co/images/homepage/glassy-orb/orb-purple.webm"
                    autoPlay
                    loop
                    muted
                    playsInline
                    className="w-full h-full object-contain"
                    style={{
                      mixBlendMode: 'screen',
                      filter: 'hue-rotate(-75deg) saturate(1.2) brightness(1.1) contrast(1.1)'
                    }}
                  />
                </div>
              </div>

            </div>
          </div>

          {/* Subtle Bottom Divider */}
          <div className="absolute bottom-0 left-1/2 -translate-x-1/2 w-full max-w-[1200px] h-px bg-gradient-to-r from-transparent via-white/10 to-transparent" />
        </section>

        {/* Product / Value Section */}
        <section className="max-w-[1000px] mx-auto px-6 py-24 text-center">
          <h2 className="font-heading text-4xl md:text-5xl font-bold mb-6 tracking-tight">Accuracy first. Clarity when it matters.</h2>
          <p className="text-xl text-zinc-400 leading-relaxed max-w-2xl mx-auto">
            Built to separate authentic voices from synthetic and cloned speech.
            We provide deep analysis of incoming audio streams, ensuring that you can trust the voice on the other end.
          </p>
        </section>

        {/* Detection Models Section */}
        <section className="max-w-[1200px] mx-auto px-6 py-16 relative">
          <div className="text-center mb-16 relative z-10">
            <h2 className="font-heading text-3xl font-bold mb-4">Detection backed by real voice models.</h2>
          </div>

          <div className="grid grid-cols-1 md:grid-cols-2 gap-8 max-w-4xl mx-auto relative z-10">
            {/* WavLM Card */}
            <div className="p-10 rounded-[2rem] bg-[#101218]/80 backdrop-blur-md border border-[#319AFF]/20 hover:border-[#319AFF]/40 transition-colors group relative overflow-hidden">
              <div className="absolute top-0 right-0 w-64 h-64 bg-[#0084FF]/5 rounded-full blur-3xl group-hover:bg-[#0084FF]/10 transition-colors pointer-events-none" />
              <div className="text-[#319AFF] text-xs font-mono font-bold tracking-widest uppercase mb-4">Core Model</div>
              <h3 className="text-3xl font-heading font-bold mb-4">WavLM</h3>
              <p className="text-zinc-400 text-lg leading-relaxed">
                Transformer-based speech representation model used for deep voice analysis and synthetic detection.
              </p>
            </div>

            {/* LFCC Card */}
            <div className="p-10 rounded-[2rem] bg-[#101218]/80 backdrop-blur-md border border-[#319AFF]/20 hover:border-[#319AFF]/40 transition-colors group relative overflow-hidden">
              <div className="absolute top-0 right-0 w-64 h-64 bg-[#0084FF]/5 rounded-full blur-3xl group-hover:bg-[#0084FF]/10 transition-colors pointer-events-none" />
              <div className="text-[#319AFF] text-xs font-mono font-bold tracking-widest uppercase mb-4">Audio Features</div>
              <h3 className="text-3xl font-heading font-bold mb-4">LFCC</h3>
              <p className="text-zinc-400 text-lg leading-relaxed">
                Linear Frequency Cepstral Coefficients precisely extracted as robust audio features for spoof and deepfake classification.
              </p>
            </div>
          </div>
        </section>

        {/* Features Section */}
        <section id="features" className="max-w-[1200px] mx-auto px-6 py-24 relative">
          <div className="text-center max-w-2xl mx-auto mb-16">
            <h2 className="font-heading text-3xl md:text-4xl font-bold mb-4">Built to Detect What Humans Can't.</h2>
          </div>

          <div className="grid grid-cols-1 md:grid-cols-3 gap-6">
            {[
              {
                icon: Activity,
                title: "Live Voice Monitoring",
                desc: "Continuously monitor and score incoming audio streams in real-time."
              },
              {
                icon: ShieldAlert,
                title: "Deepfake Detection",
                desc: "Analyze and identify fully synthetic speech generated by modern AI models."
              },
              {
                icon: Search,
                title: "File Analysis",
                desc: "Upload and inspect historical recordings for signs of manipulation or cloning."
              }
            ].map((feat, i) => (
              <div
                key={i}
                className="p-8 rounded-[1.5rem] bg-[#14161C]/40 backdrop-blur-xl border border-white/5 hover:bg-[#1a1d24]/60 transition-all duration-300 hover:-translate-y-1 group relative overflow-hidden"
              >
                <div className="w-12 h-12 rounded-2xl bg-blue-500/10 border border-blue-500/20 flex items-center justify-center mb-6 text-blue-400">
                  <feat.icon size={24} />
                </div>
                <h3 className="text-xl font-heading font-semibold mb-3">{feat.title}</h3>
                <p className="text-zinc-400 leading-relaxed">{feat.desc}</p>
              </div>
            ))}
          </div>
        </section>

        {/* How It Works */}
        <section id="how-it-works" className="max-w-[1200px] mx-auto px-6 py-24 border-t border-white/5 relative">
          <div className="text-center mb-20 relative z-10">
            <h2 className="font-heading text-3xl md:text-4xl font-bold mb-4">How Voice Integrity Works</h2>
          </div>

          <div className="grid grid-cols-1 md:grid-cols-3 gap-12 lg:gap-8 relative z-10">
            {[
              {
                step: "01",
                title: "Capture",
                desc: "Provide live audio or upload a recording."
              },
              {
                step: "02",
                title: "Analyze",
                desc: "Voice Integrity processes the audio through AI detection models."
              },
              {
                step: "03",
                title: "Detect",
                desc: "Receive a clear indication of whether the voice appears authentic or synthetic."
              }
            ].map((step, i) => (
              <div key={i} className="flex flex-col items-center text-center">
                <div className="text-[#0084FF]/30 font-heading text-6xl font-bold mb-6 tracking-tighter">
                  {step.step}
                </div>
                <h3 className="text-2xl font-heading font-semibold mb-3">{step.title}</h3>
                <p className="text-zinc-400 max-w-[280px] leading-relaxed">{step.desc}</p>
              </div>
            ))}
          </div>
        </section>

        {/* Real Product Preview */}
        <section className="max-w-[1200px] mx-auto px-6 py-32">
          <div className="flex flex-col items-center text-center mb-16">
            <div className="text-xs font-mono tracking-widest text-[#60B1FF] mb-6 uppercase">Voice Integrity Platform</div>
            <h2 className="font-heading text-4xl md:text-5xl font-bold mb-6 max-w-2xl leading-tight">
              From Voice Input to Actionable Detection.
            </h2>
            <p className="text-xl text-zinc-400 leading-relaxed max-w-2xl mx-auto mb-8">
              Analyze recordings, monitor live audio, and inspect voice integrity through the Voice Integrity platform.
            </p>
            <Link
              to="/platform"
              className="inline-flex items-center gap-2 px-8 py-4 rounded-2xl bg-white text-zinc-950 font-medium hover:bg-zinc-200 transition-colors group text-[16px]"
            >
              Open Platform <ArrowRight size={20} className="group-hover:translate-x-1 transition-transform" />
            </Link>
          </div>

          <div className="relative rounded-[2rem] p-2 bg-gradient-to-b from-white/10 to-transparent overflow-hidden shadow-[0_20px_80px_rgba(0,0,0,0.8)] border border-white/10 max-w-5xl mx-auto">
            <div className="w-full aspect-[16/9] lg:aspect-auto rounded-[1.5rem] overflow-hidden bg-[#0a0c10] relative">
              <img
                src="/sitepreview.png"
                alt="Voice Integrity Platform Screenshot"
                className="w-full h-auto object-cover"
              />
            </div>
          </div>
        </section>

        {/* Final CTA */}
        <section className="max-w-[1200px] mx-auto px-6 py-24 mb-12">
          <div className="p-16 rounded-[3rem] bg-[#101218]/80 border border-[#0084FF]/20 relative overflow-hidden flex flex-col items-center text-center">
            <div className="absolute top-0 left-1/2 -translate-x-1/2 w-[800px] h-[800px] bg-[#0084FF]/10 blur-[150px] rounded-full pointer-events-none" />

            <h2 className="font-heading text-4xl md:text-5xl lg:text-6xl font-bold mb-6 relative z-10 tracking-tight">Trust Every Voice.</h2>
            <p className="text-[20px] text-zinc-400 mb-10 relative z-10">
              Detect synthetic speech before it becomes a security risk.
            </p>

            <div className="flex flex-col sm:flex-row items-center justify-center gap-5 relative z-10 w-full sm:w-auto">
              <a
                href="https://codequantum.in/downloads/app-debug.apk"
                download
                className="w-full sm:w-auto flex items-center justify-center gap-2 px-8 py-4 rounded-[16px] bg-[#0084FF] hover:bg-[#319AFF] text-white font-medium transition-all duration-300 shadow-[0_0_20px_rgba(0,132,255,0.3)] group overflow-hidden relative"
              >
                <div className="absolute inset-0 bg-gradient-to-b from-white/20 to-transparent opacity-50 pointer-events-none" />
                <Download size={20} className="relative z-10" />
                <span className="relative z-10 text-[16px]">Download App</span>
              </a>
              <Link
                to="/platform"
                className="w-full sm:w-auto flex items-center justify-center gap-2 px-8 py-4 rounded-[16px] bg-transparent border border-white/20 hover:border-white/40 hover:bg-white/5 text-white font-medium transition-colors group"
              >
                <span className="text-[16px]">Try the Platform</span>
                <ArrowRight size={20} className="group-hover:translate-x-1 transition-transform" />
              </Link>
            </div>
          </div>
        </section>

      </main>

      {/* Footer */}
      <footer id="about" className="bg-[#08090B] py-16 relative z-10 border-t border-white/5">
        <div className="max-w-[1200px] mx-auto px-6 flex flex-col md:flex-row items-center justify-between gap-8">
          <div className="flex flex-col items-center md:items-start gap-4">
            <div className="flex items-center gap-3">
              <img src="/logo.png" alt="Voice Integrity" className="h-6 object-contain" />
              <span className="font-heading font-bold text-lg text-white">Voice Integrity</span>
            </div>
            <p className="text-zinc-500 text-sm max-w-[250px] text-center md:text-left">
              AI-powered voice deepfake and cloning detection.
            </p>
          </div>

          <div className="flex gap-8 text-sm text-zinc-400 font-medium">
            <a href="#features" className="hover:text-white transition-colors">Features</a>
            <a href="#how-it-works" className="hover:text-white transition-colors">How It Works</a>
            <Link to="/platform" className="hover:text-white transition-colors">Platform</Link>
          </div>

          <div className="text-sm text-zinc-600">
            © 2026 Voice Integrity. All rights reserved.
          </div>
        </div>
      </footer>

      <style dangerouslySetInnerHTML={{
        __html: `
        @keyframes fade-in-up {
          0% { opacity: 0; transform: translateY(30px); }
          100% { opacity: 1; transform: translateY(0); }
        }
        .animate-fade-in-up {
          animation: fade-in-up 1s cubic-bezier(0.16, 1, 0.3, 1) forwards;
        }

        @keyframes fade-in-slow {
          0% { opacity: 0; transform: scale(0.95); }
          100% { opacity: 1; transform: scale(1); }
        }
        .animate-fade-in-slow {
          animation: fade-in-slow 1.5s cubic-bezier(0.16, 1, 0.3, 1) forwards;
        }

        html { scroll-behavior: smooth; }
      `}} />
    </div>
  );
}
