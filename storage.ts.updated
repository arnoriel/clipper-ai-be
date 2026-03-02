// src/lib/storage.ts (UPDATED)
// API key OpenRouter sekarang hanya ada di backend Python
// Frontend tidak perlu menyimpan/membaca API key

import type { ViralMoment, VideoAnalysisResult } from "./AI";

export interface TextOverlay {
  id: string;
  text: string;
  x: number;
  y: number;
  fontSize: number;
  color: string;
  startSec: number | null;
  endSec: number | null;
  bold: boolean;
}

export interface ClipEdits {
  cropX: number;
  cropY: number;
  cropW: number;
  cropH: number;
  aspectRatio: "9:16" | "16:9" | "1:1" | "4:3" | "original";
  textOverlays: TextOverlay[];
  brightness: number;
  contrast: number;
  saturation: number;
  speed: number;
  trimStart: number;
  trimEnd: number;
}

export interface ProjectClip {
  momentId: string;
  moment: ViralMoment;
  edits: ClipEdits;
  exportedUrl?: string;
}

export interface Project {
  id: string;
  videoFileName: string;
  videoFileSize: number;
  videoMimeType: string;
  videoDuration: number;
  localVideoUrl?: string;
  analysisResult: VideoAnalysisResult;
  selectedClips: ProjectClip[];
  createdAt: number;
  updatedAt: number;
}

const STORAGE_KEY = "ai_clipper_projects_v2";

// ─── API Key — sekarang ada di backend, frontend tidak butuh ──────────────────
export function getApiKey(): string {
  // Kembalikan dummy string agar kode lama tidak error
  // API key sesungguhnya ada di environment variable backend
  return "server-side";
}

export function isApiKeyConfigured(): boolean {
  // Selalu true karena API key ada di backend
  return true;
}

// ─── Projects ─────────────────────────────────────────────────────────────────
export function loadProjects(): Project[] {
  try {
    const raw = localStorage.getItem(STORAGE_KEY);
    if (!raw) return [];
    const projects: Project[] = JSON.parse(raw);
    return projects.map((p) => ({ ...p, localVideoUrl: undefined }));
  } catch {
    return [];
  }
}

export function saveProject(project: Project) {
  const projects = loadProjects();
  const idx = projects.findIndex((p) => p.id === project.id);
  const toSave: Project = { ...project, localVideoUrl: undefined };
  if (idx >= 0) projects[idx] = toSave;
  else projects.unshift(toSave);
  localStorage.setItem(STORAGE_KEY, JSON.stringify(projects.slice(0, 20)));
}

export function deleteProject(id: string) {
  const projects = loadProjects().filter((p) => p.id !== id);
  localStorage.setItem(STORAGE_KEY, JSON.stringify(projects));
}

export function getProject(id: string): Project | null {
  return loadProjects().find((p) => p.id === id) || null;
}

export function defaultEdits(): ClipEdits {
  return {
    cropX: 0, cropY: 0, cropW: 1, cropH: 1,
    aspectRatio: "original",
    textOverlays: [],
    brightness: 0, contrast: 0, saturation: 0,
    speed: 1,
    trimStart: 0, trimEnd: 0,
  };
}

export function generateId(): string {
  return Math.random().toString(36).slice(2, 10);
}
