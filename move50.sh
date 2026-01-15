src="/путь/к/исходной_папке"
dst="/путь/к/папке_назначения"

ls -p "$src" | grep -v / | shuf | head -n 50 | while read f; do
  mv -- "$src/$f" "$dst/"
done
