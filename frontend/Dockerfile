# =============================================================================
# ShuttleSense Frontend Dockerfile
# Multi-stage build with Nginx for production serving
# =============================================================================

# Stage 1: Build
FROM node:20-alpine as build

WORKDIR /app

# Copy package files
COPY package*.json ./

# Install dependencies
RUN npm ci --only=production=false

# Copy source code
COPY . .

# Build for production
RUN npm run build

# Stage 2: Production with Nginx
FROM nginx:alpine as production

# Copy custom nginx config
COPY nginx.conf /etc/nginx/nginx.conf

# Copy built assets from build stage
COPY --from=build /app/dist /usr/share/nginx/html

# Add non-root user
RUN adduser -D -u 1000 shuttlesense && \
    chown -R shuttlesense:shuttlesense /usr/share/nginx/html && \
    chown -R shuttlesense:shuttlesense /var/cache/nginx && \
    touch /var/run/nginx.pid && \
    chown shuttlesense:shuttlesense /var/run/nginx.pid

# Health check
HEALTHCHECK --interval=30s --timeout=10s --start-period=5s --retries=3 \
    CMD wget -q --spider http://localhost:80/ || exit 1

EXPOSE 80

CMD ["nginx", "-g", "daemon off;"]
