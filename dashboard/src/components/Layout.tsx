import { Link, Outlet, useLocation } from 'react-router-dom';
import {
  BarChart3,
  Activity,
  Wind,
  MapPin,
  BookOpen,
  Tornado,
} from 'lucide-react';

const nav = [
  { path: '/', label: 'Model Comparison', icon: BarChart3 },
  { path: '/errors', label: 'Error Distributions', icon: Activity },
  { path: '/storms', label: 'Storm Performance', icon: Wind },
  { path: '/tracker', label: 'Storm Tracker', icon: MapPin },
  { path: '/methodology', label: 'Methodology', icon: BookOpen },
];

export function Layout() {
  const location = useLocation();

  return (
    <div className="min-h-screen flex">
      <aside className="w-64 bg-slate-900 text-slate-100 flex flex-col fixed h-full">
        <div className="p-6 border-b border-slate-700">
          <Link to="/" className="flex items-center gap-3">
            <Tornado className="w-8 h-8 text-indigo-400" />
            <span className="font-bold text-xl">Hurricane Tracker</span>
          </Link>
          <p className="text-slate-400 text-sm mt-2">Model Comparison Dashboard</p>
        </div>
        <nav className="flex-1 p-4 space-y-1">
          {nav.map(({ path, label, icon: Icon }) => (
            <Link
              key={path}
              to={path}
              className={`flex items-center gap-3 px-4 py-3 rounded-lg transition-colors ${
                location.pathname === path
                  ? 'bg-indigo-600 text-white'
                  : 'text-slate-300 hover:bg-slate-800 hover:text-white'
              }`}
            >
              <Icon className="w-5 h-5 shrink-0" />
              {label}
            </Link>
          ))}
        </nav>
        <footer className="p-4 border-t border-slate-700 text-center text-slate-500 text-sm">
          <a
            href="https://personal-site-iota-weld.vercel.app"
            target="_blank"
            rel="noopener noreferrer"
            className="text-indigo-400 hover:underline"
          >
            By Sardor Sobirov
          </a>
        </footer>
      </aside>
      <main className="flex-1 ml-64 p-8 overflow-auto">
        <Outlet />
      </main>
    </div>
  );
}
