const isProd = import.meta.env.VITE_APP_MODE === 'production';

export const API_CONFIG = {
  BACKEND_URL: (isProd ? import.meta.env.VITE_PROD_BACKEND_URL : import.meta.env.VITE_DEV_BACKEND_URL) || 'http://127.0.0.1:8000',
  BACKEND_WS: (isProd ? import.meta.env.VITE_PROD_BACKEND_WS : import.meta.env.VITE_DEV_BACKEND_WS) || 'ws://127.0.0.1:8000/ws',
  RELAY_URL: (isProd ? import.meta.env.VITE_PROD_RELAY_URL : import.meta.env.VITE_DEV_RELAY_URL) || 'http://127.0.0.1:8001',
  RELAY_CALLER_WS: (isProd ? import.meta.env.VITE_PROD_RELAY_CALLER_WS : import.meta.env.VITE_DEV_RELAY_CALLER_WS) || 'ws://127.0.0.1:8001/ws/caller',
  RELAY_RECEIVER_WS: (isProd ? import.meta.env.VITE_PROD_RELAY_RECEIVER_WS : import.meta.env.VITE_DEV_RELAY_RECEIVER_WS) || 'ws://127.0.0.1:8001/ws/receiver',
};
