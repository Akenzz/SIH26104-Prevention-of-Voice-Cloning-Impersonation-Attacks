import { Outlet, NavLink } from 'react-router-dom';
import { LayoutDashboard, Radio, FileAudio, Settings as SettingsIcon, ShieldCheck } from 'lucide-react';

const navItems = [
  { path: '/', label: 'Overview', icon: LayoutDashboard },
  { path: '/live-monitor', label: 'Live Monitor', icon: Radio },
  { path: '/file-analysis', label: 'File Analysis', icon: FileAudio },
  { path: '/settings', label: 'Settings', icon: SettingsIcon },
];

export default function Layout() {
  return (
    <div className="flex h-[100dvh] w-screen overflow-hidden bg-zinc-950 text-zinc-100 flex-col md:flex-row">
      {/* Sidebar for Desktop */}
      <aside className="hidden md:flex w-64 border-r border-zinc-800 bg-zinc-900/50 flex-col">
        <div className="p-6 flex items-center gap-3 border-b border-zinc-800">
          <div className="w-8 h-8 rounded-md bg-zinc-100 flex items-center justify-center text-zinc-950">
            <ShieldCheck size={20} strokeWidth={2.5} />
          </div>
          <h1 className="font-semibold text-lg tracking-tight text-zinc-100">Voice Integrity</h1>
        </div>
        
        <nav className="flex-1 p-4 space-y-1">
          {navItems.map((item) => (
            <NavLink
              key={item.path}
              to={item.path}
              className={({ isActive }) =>
                `flex items-center gap-3 px-3 py-2.5 rounded-md text-sm font-medium transition-colors ${
                  isActive
                    ? 'bg-zinc-800 text-zinc-100'
                    : 'text-zinc-400 hover:bg-zinc-800/50 hover:text-zinc-200'
                }`
              }
            >
              <item.icon size={18} />
              {item.label}
            </NavLink>
          ))}
        </nav>
      </aside>

      {/* Main Content Area */}
      <main className="flex-1 flex flex-col overflow-hidden relative">
        <div className="flex-1 overflow-auto bg-zinc-950 pb-16 md:pb-0">
          <Outlet />
        </div>
      </main>

      {/* Bottom Nav for Mobile */}
      <nav className="md:hidden fixed bottom-0 left-0 right-0 bg-zinc-900 border-t border-zinc-800 flex justify-around p-2 pb-safe z-50">
        {navItems.map((item) => (
          <NavLink
            key={item.path}
            to={item.path}
            className={({ isActive }) =>
              `flex flex-col items-center gap-1 p-2 rounded-lg text-xs font-medium transition-colors ${
                isActive
                  ? 'text-zinc-100'
                  : 'text-zinc-500 hover:text-zinc-300'
              }`
            }
          >
            <item.icon size={20} />
            <span className="hidden sm:inline">{item.label}</span>
          </NavLink>
        ))}
      </nav>
    </div>
  );
}
