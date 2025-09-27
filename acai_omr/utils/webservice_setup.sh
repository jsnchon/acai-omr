#!/bin/bash

# USAGE
# this assumes the poetry venv has already been set up 
# positional args: [absolute path to project root directory] [root domain name]

APP_NAME="acai-omr"
SOCKET_FILE_PATH="/run/$APP_NAME.sock"

root_dir=$1
root_domain=$2
echo "Project root directory: $root_dir" 
echo "Root domain to use for dns: $root_domain"

venv_path=$(poetry env info --path)
if [[ -z "$venv_path" ]]; then
    echo "poetry env returned no path. Make sure the poetry project has been set up. Aborting"
    exit 1
else
    echo "Poetry venv path: $venv_path"
fi

echo "Installing apt dependencies"
sudo apt update && sudo apt upgrade -y
sudo apt install nginx && sudo apt install musescore3 && sudo apt install imagemagick && sudo apt install certbot python3-certbot-nginx -y

# include /usr/bin in environment PATH for musescore/imagemagick cli and use gevent workers for SSE support
echo "Creating gunicorn service file"
gunicorn_file_path="/etc/systemd/system/$APP_NAME.service"
sudo tee "$gunicorn_file_path" << EOF
[Unit]
Description=Gunicorn worker to serve Flask app
After=network.target

[Service]
User=root
Group=www-data
WorkingDirectory=$root_dir
Environment="PATH=$venv_path/bin/:/usr/bin/" 
ExecStart=$venv_path/bin/gunicorn --timeout 0 --worker-class gevent --workers 1 --bind unix:$SOCKET_FILE_PATH -m 007 "acai_omr.wsgi:app" 

[Install]
WantedBy=multi-user.target
EOF

echo "Starting gunicorn service"
sudo systemctl daemon-reload
sudo systemctl start "$APP_NAME"
sudo systemctl enable "$APP_NAME"

# specify streaming endpoint settings to allow SSE support (eg prevent stream buffering)
echo "Creating nginx file"
nginx_file_path="/etc/nginx/sites-available/$APP_NAME"
sudo tee "$nginx_file_path" << EOF
server {
    listen 80;
    server_name $root_domain www.$root_domain;
    client_max_body_size 50M;

    location / {
        include proxy_params;
        proxy_pass http://unix:$SOCKET_FILE_PATH;
    }
    
    location /inference/stream {
        include proxy_params;
        proxy_pass http://unix:$SOCKET_FILE_PATH;

        proxy_buffering off;
        proxy_cache off;
        proxy_set_header Connection '';
        chunked_transfer_encoding off;
        proxy_read_timeout 60s;
    }
}
EOF

echo "Starting nginx"
sudo ln -s "$nginx_file_path" /etc/nginx/sites-enabled
sudo systemctl daemon-reload
sudo systemctl restart nginx
sudo systemctl enable nginx

echo "Adding https support"
sudo certbot --nginx -d "$root_domain" -d www."$root_domain"

echo "Setup complete. App should be running at https://$root_domain"
