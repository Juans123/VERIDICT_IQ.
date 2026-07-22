# Guía para subir VERIDICT IQ a GitHub

## Opción A: desde la página de GitHub

1. Inicie sesión en GitHub.
2. Seleccione **New repository**.
3. Nombre sugerido: `VERIDICT_IQ`.
4. Seleccione **Private** para mantener controlado el acceso.
5. No marque la creación automática de README, `.gitignore` o licencia, porque ya están incluidos.
6. Cree el repositorio.
7. En la página vacía, seleccione **uploading an existing file**.
8. Arrastre todo el contenido de esta carpeta, no la carpeta contenedora completa.
9. Use el mensaje de commit: `feat: publish S5 preliminary results repository`.
10. Confirme con **Commit changes**.

## Opción B: desde Git Bash o la terminal de VS Code

Desde esta carpeta:

```bash
git init
git add .
git commit -m "feat: publish S5 preliminary results repository"
git branch -M main
git remote add origin https://github.com/USUARIO/VERIDICT_IQ.git
git push -u origin main
```

## Crear la versión de la entrega

```bash
git tag -a v0.1-resultados-preliminares -m "Entrega S5: informe de resultados preliminares"
git push origin v0.1-resultados-preliminares
```

## Dar acceso al profesor en un repositorio privado

1. Abra **Settings** del repositorio.
2. Entre en **Collaborators**.
3. Seleccione **Add people**.
4. Ingrese el usuario de GitHub o correo del profesor.
5. Envíe la invitación.

## Verificación antes de compartir

- El README se visualiza correctamente.
- El dataset publicado está en `data/synthetic/`.
- No existen archivos reales en `data/raw/`, `data/interim/` o `data/processed/`.
- No se incluyeron credenciales ni archivos `.env`.
- El notebook abre correctamente.
- `pytest -q` finaliza sin errores.
- `python run_demo.py` genera artefactos en `artifacts/latest/`.
