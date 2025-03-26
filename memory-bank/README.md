# Memory Bank: Diver Detection System

## Purpose
The Memory Bank serves as the central knowledge repository for the Diver Detection System project. It maintains all critical information about the project's goals, architecture, technical aspects, and current progress.

## Core Files

| File | Description |
|------|-------------|
| [projectbrief.md](./projectbrief.md) | Foundation document that defines core requirements and goals |
| [productContext.md](./productContext.md) | Why this project exists and problems it solves |
| [systemPatterns.md](./systemPatterns.md) | System architecture and design patterns |
| [techContext.md](./techContext.md) | Technologies used and technical constraints |
| [activeContext.md](./activeContext.md) | Current work focus and immediate next steps |
| [progress.md](./progress.md) | Project status, what works, and what's left to build |
| [.clinerules](./.clinerules) | Project-specific patterns and intelligence |

## Structure
The Memory Bank follows a hierarchical structure where files build upon each other:

```
projectbrief.md
  ├── productContext.md
  ├── systemPatterns.md 
  └── techContext.md
       └── activeContext.md
            └── progress.md
```

## Usage
- Reference these files at the start of any work session
- Update files when significant progress is made
- Focus on activeContext.md and progress.md for current status
- Use .clinerules to understand project-specific patterns

## Current Project Status
- Development environment successfully set up
- Container with GPU support and X11 forwarding configured
- YOLO framework tested and functioning
- Ready to begin dataset collection and model training 