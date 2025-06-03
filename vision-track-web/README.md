# Contenido del archivo README.md

# Vision Track Web

Este proyecto es una aplicación web desarrollada con Flask que permite la interacción con la API de RENIEC y la funcionalidad de reconocimiento facial. 

## Estructura del Proyecto

- `src/app.py`: Punto de entrada de la aplicación Flask.
- `src/api/`: Módulo que maneja la interacción con la API de RENIEC.
- `src/facial/`: Módulo que implementa la lógica de reconocimiento facial.
- `src/static/`: Archivos estáticos como CSS y JavaScript.
- `src/templates/`: Plantillas HTML para la interfaz de usuario.
- `src/utils/`: Utilidades y configuraciones del proyecto.
- `known_faces/`: Carpeta para almacenar imágenes de rostros conocidos.
- `requirements.txt`: Dependencias necesarias para el proyecto.

## Instalación

1. Clona el repositorio:
   ```bash
   git clone <url-del-repositorio>
   cd vision-track-web
   ```

2. Instala las dependencias:
   ```bash
   pip install -r requirements.txt
   ```

## Ejecución

Para ejecutar la aplicación, utiliza el siguiente comando:

```bash
python src/app.py
```

La aplicación estará disponible en `http://localhost:5000`.

## Contribuciones

Las contribuciones son bienvenidas. Si deseas contribuir, por favor abre un issue o envía un pull request.

## Licencia

Este proyecto está bajo la Licencia MIT.