import { mkdirSync } from 'fs';
import { appendFileSync } from 'fs';
import path from 'path';

mkdirSync('logs', { recursive: true });

const logFile = path.join('logs', `${new Date().toISOString().replace(/[:.]/g, '-')}.log`);

type LogLevel = 'DEBUG' | 'INFO' | 'WARN' | 'ERROR';

function formatMessage(level: LogLevel, msg: string): string {
  const ts = new Date().toISOString();
  return `${ts} [${level}] ${msg}`;
}

function log(level: LogLevel, msg: string): void {
  const formatted = formatMessage(level, msg);
  if (level === 'ERROR') console.error(formatted);
  else if (level === 'WARN') console.warn(formatted);
  else console.log(formatted);

  try {
    appendFileSync(logFile, formatted + '\n');
  } catch {}
}

export const logger = {
  debug: (msg: string) => log('DEBUG', msg),
  info: (msg: string) => log('INFO', msg),
  warn: (msg: string) => log('WARN', msg),
  error: (msg: string) => log('ERROR', msg),
};
