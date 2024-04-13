# Зависимости
Для корректрой работы приложения необходимо установить и настроить следующие утилиты.

## GitHub CLI
GitHub CLI требуется для получения токена авторизации и дальнейшей его передачи в запускаемые GitHub Actions. Токен автороизации используется в реализованных GitHub Actions для получения информации о открытых пулл-реквестах и получения их содержимого.

### Установка
Для установки GitHub CLI в Arch Linux воспользуйтесь следующей командой:    
```sh
sudo pamac install github-cli
```

Для установки GitHub CLI в Ubuntu Linux воспользуйтесь следующей командой:  
```sh
sudo mkdir -p -m 755 /etc/apt/keyrings && wget -qO- https://cli.github.com/packages/githubcli-archive-keyring.gpg | sudo tee /etc/apt/keyrings/githubcli-archive-keyring.gpg > /dev/null \
&& sudo chmod go+r /etc/apt/keyrings/githubcli-archive-keyring.gpg \
&& echo "deb [arch=$(dpkg --print-architecture) signed-by=/etc/apt/keyrings/githubcli-archive-keyring.gpg] https://cli.github.com/packages stable main" | sudo tee /etc/apt/sources.list.d/github-cli.list > /dev/null \
&& sudo apt update \
&& sudo apt install gh -y
```

[Оффициальная документация по установке GitHub CLI.](https://github.com/cli/cli/blob/trunk/docs/install_linux.md)

Проверьте корректность установки с помощью команды:  
```sh
gh --version
```

Пример корректного отклика:  
```sh
gh version 2.45.0 (2024-03-04)
https://github.com/cli/cli/releases/tag/v2.45.0
```

### Настройка
Для дальнейшей работы с GitHub CLI необходимо провести аутентификацию. Для прохождения аутентификации в GitHub CLI воспользуйтесь следующей командой:  
```sh
gh auth login
```

Удостоверьтесь в корректном прохождении аутентификации:
```sh
gh auth token
```

В результате выполения команды выше Вы должны получить токен аутентификации.

> #### Примечание
> Если Вам никак не удается установить GitHub CLI или авторизироваться, то Вы можете создать токен аутентификации вручную. Для этого перейдите по следующей [ссылке](https://github.com/settings/tokens). В правом верхнем углу выберите `Generate new token`. При генерации токена установите для него доступ к репозиториям. Сохраните полученный токен.

## SSH
Приложение использует SSH для получения содержимого пулл-реквестов.

> #### Примечание
> В процессе разработки принято решение использовать rsa SSH-ключи, так как данные ключи поддерживаются даже на устаревших системах. Так как приложение работает с использованием GitHub Actions, оно не имеет возможности напрямую получить доступ ко всем SSH ключам, используемым в системе, так как по изначальной задумке GitHub Actions должны запускаться не локально, а на удаленной виртуальной системе. Поэтому указание SSH ключей происходит напрямую при запуске рабочего процесса GitHub Actions, и это должны быть именно rsa ключи.

### Генерация ключей
Для генерации rsa ключей используйте следующую команду:  
```sh
ssh-keygen -t rsa
```

Для проверки сгенерирванных ключей проверьте содержимое файлов `~/.ssh/id_rsa.pub` (публичный ключ) и `~/.ssh/id_rsa` (приватный ключ).

### Добавление ключей в GitHub аккаунт
Перейдите по [ссылке](https://github.com/settings/keys). В правом верхнем углу выберите `New SSH key`. В поле `Key` укажите содержимое файла `~/.ssh/id_rsa.pub`.

Для проверки наличия SSH-соединения воспользуйтесь следующей командой:
```sh
ssh -T git@github.com
```

Результат вывода должен быть следующим:
```
Hi <Ваше имя пользователя>! You've successfully authenticated, but GitHub does not provide shell access.
```

[Подробная информация о установке SSH-соединения с GitHub](https://docs.github.com/en/authentication/connecting-to-github-with-ssh/testing-your-ssh-connection).
